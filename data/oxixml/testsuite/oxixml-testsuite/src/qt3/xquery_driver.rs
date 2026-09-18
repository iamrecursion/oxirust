//! The QT3 catalogue driven against [`oxixml_xquery`].
//!
//! Every `<test-case>` whose dependencies name only an XQuery (`XQ…`) specification version and
//! a feature this engine implements is compiled as XQuery 3.1 and executed; the outcome is
//! checked with [`crate::assertions::check_with_serialization`], whose `serialization-matches`
//! is decided against the output declarations the query's own prolog carries (see the
//! `serialization` module beside this one).
//!
//! **Scope**: this driver reproduces the environment (sources, namespaces, static base URI,
//! parameters, collections, decimal formats, `<source role="$name">` variables) exactly as
//! [`super::xpath_driver`] does, and additionally resolves `moduleImport` by reading each test
//! case's own `<module uri file>` children into an in-memory module library (see
//! `read_modules`). It does not resolve a `<test file="...">` case whose query body itself
//! lives in an external file, nor a custom UCA collation — cases needing either are skipped as an
//! unsupported dependency/environment, not attempted.
//!
//! **Version pinning**: unlike [`super::xpath_driver`], `oxixml_xquery::XQuery::compile` has no
//! per-version compile mode — this engine implements XQuery 3.1 throughout (the same "3.0+" core
//! `query/oxixml-xquery/tests/qt3.rs`'s own gate documents). A case pinned to versions that do not
//! include 3.1 is exercising behaviour this engine does not implement at all (most visibly XQuery
//! 1.0's *static* circularity check, relaxed to a dynamic one from 3.0 on, and the smaller
//! pre-3.0 reserved-function-name list), not behaviour it gets wrong, and is skipped rather than
//! attempted — the same honest "not attempted" this driver gives an unsupported `<dependency
//! type="feature">`. That decision belongs to the shared `spec_satisfied` rule, which the XPath
//! driver applies too: a `spec` list is satisfied only when one of its alternatives names 3.1, so
//! `value="XQ10"` and `value="XQ10 XQ30"` are skipped while `value="XP20 XP30 XP31 XQ10 XQ30
//! XQ31"` — which names 3.1 among its alternatives — is run, at 3.1, as the catalogue intends.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use oxixml_fn::{CollectingTrace, FnContext};
use oxixml_infoset::{NodeRef, Tree, XdmNode};
use oxixml_xpath::{StaticContext, XPath, XPathVersion};
use oxixml_xquery::{
    DynamicQueryContext, InMemoryModules, StaticQueryContext, XQuery, XQueryError,
};

use super::budget::{CaseOutcome, CaseRunner, WORKER_STACK, progress, timeout_message};
use super::serialization;
use super::{
    Environment, apply_default_base_uri, dependencies_supported, parameter_name, parse_file,
    qt3_root, read_environment, resolve_environment, shared_environments, unavailable_reason,
};
use crate::assertions::{ErrorInfo, Verdict, check_with_serialization};
use crate::report::SuiteReport;
use crate::xdm_util::{attribute, children_named, element_children, find_child};

/// The features of the suite this engine implements; see the module documentation for what is
/// deliberately not on this list.
const SUPPORTED_FEATURES: &[&str] = &[
    "higherOrderFunctions",
    // A document parsed by `oxixml_infoset::Tree` carries the infoset properties its DTD
    // declares: an `<!ATTLIST>` of type `ID`, `IDREF` or `IDREFS` reaches the XDM attribute
    // node's `is-id`/`is-idrefs` properties, which is exactly what `fn:id` and `fn:idref` read.
    // The 32 cases that gate on this — `fn/id.xml`, `fn/idref.xml` and one
    // `app/FunctxFunctx.xml` case that is XQuery-only — all run against fixtures with an
    // *internal* subset (`fn/id/iddtd.xml`, `app/FunctxFn/functx_book.xml`).
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

/// Runs the whole catalogue against the XQuery engine.
#[must_use]
pub fn run() -> SuiteReport {
    let name = super::suite_name("qt3-xquery");
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
    progress("qt3-xquery", &set_name);

    let mut environments = shared.clone();
    environments.entry("empty".to_owned()).or_default();
    for definition in children_named(&root, "environment") {
        if let Some(name) = attribute(&definition, "name") {
            environments.insert(name, read_environment(&definition));
        }
    }
    let set_supported = dependencies_supported(&root, "XQ", SUPPORTED_FEATURES);

    for case in children_named(&root, "test-case") {
        let case_name = attribute(&case, "name").unwrap_or_default();
        let id = format!("{set_name}/{case_name}");
        if !set_supported || !dependencies_supported(&case, "XQ", SUPPORTED_FEATURES) {
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
        let query = test.string_value();
        if query.trim().is_empty() {
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

        let modules = read_modules(&case, directory);
        // A `serialization-matches` is decided against the query's *own* serialization options,
        // which `oxixml_xquery` records while compiling but leaves to its caller to apply — this
        // driver is that caller, so it reads the prolog's output declarations back out of the
        // query text. `None` means they name a serialization this harness cannot reproduce, and
        // the assertion is then skipped rather than judged.
        let serialization = serialization::declared_parameters(&query);
        // Kept for the skip label below, because the parameters themselves move into the job.
        let serialization_reproduced = serialization.is_some();
        // The whole case — compilation, evaluation *and* assertion checking — runs on the
        // runner's worker thread under a wall-clock budget; nothing but the verdict crosses back.
        let directory = directory.to_path_buf();
        let body = query.clone();
        let job = Box::new(move || {
            // The job runs on the case runner's worker, whose stack is `WORKER_STACK`; the
            // engine's own guard has to be told so, or it measures a query against the 2 MiB it
            // assumes by default and stops a case the thread has ample room for.
            evaluate_case(&body, &environment, &directory, modules, WORKER_STACK).map(|outcome| {
                let verdict =
                    check_with_serialization(&assertion, &outcome, serialization.as_ref());
                // A `validate` expression (or typed `import schema`) needs the Schema Import
                // Feature, which this basic processor declines statically with `XQST0075`. When
                // the case wanted a value — so the verdict is a failure — that is a scope gap,
                // not a conformance failure, so it is skipped; a case that actually *expected*
                // `XQST0075` has already passed above and is left untouched.
                match (&verdict, &outcome) {
                    (Verdict::Fail(_), Err(error)) if is_schema_feature_error(&error.code) => {
                        Verdict::Skip
                    }
                    _ => verdict,
                }
            })
        });
        // Read before the run; see `CaseRunner::still_running`.
        let contention = runner.still_running();
        let abandoned = runner.abandoned();
        let Some(verdict) = runner.run(job) else {
            let message = timeout_message(runner.budget(), contention, abandoned, &query);
            report.record_fail(&id, message);
            continue;
        };
        match verdict {
            None => report.record_skip("environment could not be reproduced"),
            Some(Verdict::Pass) => report.record_pass(),
            Some(Verdict::Fail(reason)) => report.record_fail(&id, reason),
            // A case whose serialization could not be *reproduced* is a different gap from an
            // assertion kind this checker does not know, and one label for both hides which of
            // the two the residue is made of. `declared_parameters` answering `None` means the
            // text the assertion is about was never produced — an output method or parameter
            // `oxixml_fn::xmlio` declines — while the assertion itself was perfectly well
            // understood.
            Some(Verdict::Skip) if !serialization_reproduced => {
                report.record_skip("serialization could not be reproduced");
            }
            Some(Verdict::Skip) => report.record_skip("unsupported assertion kind"),
        }
    }
}

/// Reads a test-case's own `<module uri="..." file="...">` children — library modules an `import
/// module` inside the query may reach — into an [`InMemoryModules`] resolver.
///
/// `None` when the case declares no `<module>` at all, so [`build_context`] installs no resolver
/// rather than an empty one: the two are equivalent in outcome (an unresolvable `import module`
/// raises `XQST0059` either way), but installing nothing is the more honest reflection of what
/// this case actually needs.
fn read_modules(case: &NodeRef, directory: &Path) -> Option<InMemoryModules> {
    let mut modules = InMemoryModules::new();
    let mut any = false;
    for module in children_named(case, "module") {
        let (Some(target_namespace), Some(file)) =
            (attribute(&module, "uri"), attribute(&module, "file"))
        else {
            continue;
        };
        let Ok(source) = std::fs::read_to_string(directory.join(&file)) else {
            continue;
        };
        modules.insert(target_namespace, source);
        any = true;
    }
    any.then_some(modules)
}

struct Prepared {
    statik: StaticQueryContext,
    context_item: Option<NodeRef>,
    parameters: Vec<(String, String)>,
    source_variables: Vec<(String, NodeRef)>,
    documents: Arc<super::Documents>,
    trees: Vec<Tree>,
}

fn build_context(
    environment: &Environment,
    directory: &Path,
    modules: Option<InMemoryModules>,
) -> Option<Prepared> {
    let resolved = resolve_environment(environment, directory)?;
    let mut statik = StaticQueryContext::new();
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
    if let Some(modules) = modules {
        statik = statik.with_module_resolver(Arc::new(modules));
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

/// Compiles and runs one case's query.
///
/// `stack_size` is how much native stack the *calling thread* actually has. The engine bounds a
/// recursive query against a declared stack rather than a measured one (there is nothing in
/// `std` that reports what is left), and its default is
/// [`oxixml_xquery::DEFAULT_STACK_SIZE`] — 2 MiB, the stack an ordinary spawned thread gets.
/// The catalogue contains queries that recurse thousands of levels deep on purpose, which is why
/// cases run on a worker with a much larger stack; passing that size through is what stops the
/// engine from reporting `XPDY0130` for a query the thread has room for. Passing more than the
/// thread really has would trade a bounded error for a process abort, so this is a parameter and
/// not a constant.
fn evaluate_case(
    query: &str,
    environment: &Environment,
    directory: &Path,
    modules: Option<InMemoryModules>,
    stack_size: usize,
) -> Option<Result<oxixml_xpath::XpathValue, ErrorInfo>> {
    let Prepared {
        statik,
        context_item,
        parameters,
        source_variables,
        documents,
        trees,
    } = build_context(environment, directory, modules)?;

    // Parameter `select` expressions share the query's namespace bindings but are themselves
    // plain XPath, evaluated once up front (the suite's `<param>` values do not reference the
    // query's own variables).
    let mut param_context = StaticContext::new();
    for (prefix, uri) in statik.namespaces() {
        param_context = param_context.with_namespace(prefix.to_owned(), uri.to_owned());
    }

    let compiled = match XQuery::compile(query, statik) {
        Ok(compiled) => compiled,
        Err(error) => return Some(Err(normalize_error(&error))),
    };
    let fn_context = FnContext::new().with_trace_sink(Arc::new(CollectingTrace::new()));
    let mut context = DynamicQueryContext::new()
        .with_fn_context(fn_context)
        .with_stack_size(stack_size)
        .with_document_resolver(documents);
    for (name, format) in &environment.decimal_formats {
        context = context.with_decimal_format(name.clone(), format.clone());
    }
    if let Some(item) = context_item {
        context = context.with_context_item(item);
    }
    // An explicit `<context-item select="…">` supersedes a `<source role=".">`: it is the general
    // form, able to name an atomic value or an array where a source can only be a document.
    if let Some(select) = &environment.context_item_select {
        match evaluate_selector(select, &param_context) {
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
        let bound = match evaluate_selector(select, &param_context) {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        context = context.with_variable_name(parameter_name(name, &environment.namespaces), bound);
    }
    let outcome = compiled
        .execute_value(&context)
        .map_err(|error| normalize_error(&error));
    drop(trees);
    Some(outcome)
}

/// Evaluates one `<param select="…">` or `<context-item select="…">` expression: plain XPath in
/// the case's own namespaces, with no focus and no variables of its own (the catalogue's selectors
/// are literals and simple constructor calls, never references to the case's other bindings).
fn evaluate_selector(
    select: &str,
    statik: &StaticContext,
) -> Result<oxixml_xpath::XpathValue, ErrorInfo> {
    XPath::compile(select, XPathVersion::XPath31, statik.clone())
        .and_then(|compiled| {
            let inner = compiled.context();
            compiled.evaluate_value_in(&inner)
        })
        .map_err(|error| {
            ErrorInfo::qualified(
                error.name().namespace(),
                error.name().local_name(),
                error.to_string(),
            )
        })
}

fn normalize_error(error: &XQueryError) -> ErrorInfo {
    ErrorInfo::qualified(
        error.name().namespace(),
        error.name().local_name(),
        error.to_string(),
    )
}

/// Whether an error code marks a feature this basic processor deliberately does not implement —
/// the Schema Import/Validation Feature, which a `validate` expression (or a typed `import
/// schema`) needs and which the engine reports statically as `XQST0075`. A case that requires it
/// is outside this processor's scope and is skipped rather than counted as a conformance failure.
fn is_schema_feature_error(code: &str) -> bool {
    code == "XQST0075"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stack a unit test declares: the engine's own default, which is what an ordinary
    /// spawned thread — the one a test body runs on — actually has. Never [`WORKER_STACK`]:
    /// declaring a stack the thread does not have turns a bounded `XPDY0130` into an abort.
    const TEST_STACK: usize = oxixml_xquery::DEFAULT_STACK_SIZE;

    fn evaluate(
        query: &str,
        environment: &Environment,
    ) -> Option<Result<oxixml_xpath::XpathValue, ErrorInfo>> {
        evaluate_case(query, environment, Path::new("."), None, TEST_STACK)
    }

    #[test]
    fn evaluate_case_runs_a_simple_query_without_an_environment() {
        let environment = Environment::default();
        let outcome = evaluate("1 + 1", &environment);
        let value = outcome
            .expect("no environment reproduction is needed")
            .expect("no compile/execute error");
        assert_eq!(value.to_string(), "2");
    }

    #[test]
    fn evaluate_case_reports_a_static_error() {
        let environment = Environment::default();
        let outcome =
            evaluate("let $x := ", &environment).expect("no environment reproduction is needed");
        assert!(outcome.is_err());
    }

    /// `prod-TryCatchExpr/try-catch-fn-error-1` raises
    /// `Q{http://www.example.com/}EXER3141`; the catalogue writes the expected code as that same
    /// `EQName`, so the driver has to hand the checker the namespace and not just the local name.
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
    }

    /// A standard code stays in the bare spelling every other catalogue entry is written in.
    #[test]
    fn a_standard_error_keeps_its_bare_spelling() {
        let environment = Environment::default();
        let error = evaluate("1 idiv 0", &environment)
            .expect("no environment reproduction is needed")
            .expect_err("division by zero");
        assert_eq!(error.code, "FOAR0001");
    }

    /// The stack budget is passed through, not switched off: told only what an ordinary thread
    /// has, the engine must still stop a runaway recursion with a bounded `XPDY0130` rather than
    /// let it take the process down.
    #[test]
    fn a_runaway_recursion_is_still_bounded_by_the_declared_stack() {
        let environment = Environment::default();
        let error = evaluate(
            r#"declare function local:deep($n as xs:integer) as xs:integer {
                   if ($n eq 0) then 0 else local:deep($n - 1) + 1
               };
               local:deep(1000000)"#,
            &environment,
        )
        .expect("no environment reproduction is needed")
        .expect_err("a million levels is past any thread's stack");
        assert_eq!(error.code, "XPDY0130", "{error:?}");
    }

    #[test]
    fn a_schema_environment_is_not_reproduced() {
        let environment = Environment {
            needs_schema: true,
            ..Environment::default()
        };
        assert!(evaluate("1", &environment).is_none());
    }

    /// The two-case shape `prod/FunctionDecl.xml` actually uses for
    /// `function-decl-reserved-function-names-001`/`-002`: the exact-`XQ10` twin names a version
    /// this engine does not implement and must not be attempted; the open-ended-`XQ30+` twin (this
    /// engine's own 3.1 satisfies "3.0 or later") is. And the six-alternative shape
    /// `fn/concat.xml` writes — old exact versions *and* `XQ31` — names 3.1 among its
    /// alternatives, so it is attempted rather than dismissed for the company it keeps.
    #[test]
    fn only_a_pin_that_excludes_31_is_unsupported() {
        for (value, attempted) in [
            ("XQ10", false),
            ("XQ10 XQ30", false),
            ("XQ30+", true),
            ("XQ31", true),
            ("XP20 XP30 XP31 XQ10 XQ30 XQ31", true),
        ] {
            let tree = Tree::parse_str(&format!(
                r#"<test-case><dependency type="spec" value="{value}"/></test-case>"#
            ))
            .expect("well-formed fixture");
            let case = tree.document_element().expect("a document element");
            assert_eq!(
                dependencies_supported(&case, "XQ", SUPPORTED_FEATURES),
                attempted,
                "{value}"
            );
        }
    }

    /// The XQuery half of the `infoset-dtd` declaration (see the XPath driver's own test for the
    /// reasoning): thirty-two cases gate on it, and the one this reproduces is the shape
    /// `app/FunctxFunctx.xml`'s `functx-functx-id-from-element-1` uses — an attribute reached
    /// through `id(.) is ..`, which is true only when the `<!ATTLIST>`-declared `ID` reached the
    /// XDM attribute node.
    #[test]
    fn a_dtd_declared_id_is_found_by_fn_id() {
        let case = Tree::parse_str(
            r#"<test-case><dependency type="feature" value="infoset-dtd"/></test-case>"#,
        )
        .expect("well-formed fixture");
        let case = case.document_element().expect("a document element");
        assert!(dependencies_supported(&case, "XQ", SUPPORTED_FEATURES));

        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xquery-infoset-dtd-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(
            directory.join("book.xml"),
            concat!(
                "<!DOCTYPE book [\n",
                "<!ELEMENT book (section*)>\n",
                "<!ELEMENT section (#PCDATA)>\n",
                "<!ATTLIST section id ID #REQUIRED>\n",
                "]>\n",
                "<book><section id=\"preface\">text</section></book>",
            ),
        )
        .expect("fixture file");
        let environment = Environment {
            sources: vec![crate::qt3::Source {
                role: ".".to_owned(),
                file: "book.xml".to_owned(),
                uri: None,
            }],
            ..Environment::default()
        };

        let value = evaluate_case(
            "let $book := (/) return data(($book/book/section[1]/@*[id(.) is ..])[1])",
            &environment,
            &directory,
            None,
            TEST_STACK,
        )
        .expect("the environment reproduces")
        .expect("no compile/execute error");
        assert_eq!(value.to_string(), "preface");
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
    fn a_module_import_resolves_through_a_case_declared_module() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xquery-module-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(
            directory.join("lib.xq"),
            r#"module namespace test1="http://example.com/test1";
               declare function test1:ok() { "ok" };"#,
        )
        .expect("fixture module file");
        let tree = Tree::parse_str(
            r#"<test-case>
                <module uri="http://example.com/test1" file="lib.xq"/>
            </test-case>"#,
        )
        .expect("well-formed fixture");
        let case = tree.document_element().expect("a document element");
        let modules = read_modules(&case, &directory).expect("one <module> child");
        let environment = Environment::default();
        let outcome = evaluate_case(
            r#"import module namespace test1="http://example.com/test1";
               test1:ok()"#,
            &environment,
            &directory,
            Some(modules),
            TEST_STACK,
        )
        .expect("no environment reproduction is needed")
        .expect("the import resolves");
        assert_eq!(outcome.to_string(), "ok");
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn without_a_module_declaration_an_import_is_unresolvable() {
        let environment = Environment::default();
        let outcome = evaluate(
            r#"import module namespace test1="http://example.com/test1";
               test1:ok()"#,
            &environment,
        )
        .expect("no environment reproduction is needed");
        assert!(outcome.is_err(), "nothing resolves the import");
    }

    #[test]
    fn a_dollar_role_source_is_reachable_as_a_bound_variable() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xquery-role-var-{}-{}",
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
            None,
            TEST_STACK,
        )
        .expect("the source reproduces")
        .expect("no evaluation error");
        assert_eq!(outcome.to_string(), "1");
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
