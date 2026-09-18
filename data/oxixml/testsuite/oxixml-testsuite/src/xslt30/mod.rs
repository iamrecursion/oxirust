//! W3C XSLT 3.0 test suite (`xslt30-test`) runner over the **whole** catalogue.
//!
//! Every `<test-set>` the catalogue lists is enumerated and every `<test-case>` in it reaches a
//! disposition: a pass, a failure, or a skip carrying a reason that names what stopped it. No test
//! set is excluded, because an excluded set leaves no trace in the tally at all — unlike a
//! labelled skip, which is a blind spot one can count and act on. A run's `total` is therefore the
//! size of the catalogue, and the proportion of it this processor adjudicates is a number the
//! report carries rather than a choice made before the run starts.
//!
//! Which cases are *out of scope* is the catalogue's own decision, expressed by the
//! `<dependencies>` a case or its test set declares and answered here against this module's
//! `UNSUPPORTED_FEATURES`, `SUPPORTED_FEATURES` and the capability answers beside them. A
//! dependency written on the `<test-set>` applies to every case in it — the catalogue schema says
//! so in as many words ("A dependency at the level of a test-set applies to all test cases in that
//! test-set") and the suite's own reference runner reads exactly the union, `(.|..)/dependencies`
//! — which is what turns the ~2 900 streaming cases and the ~1 100 XSLT-2.0-only cases into named,
//! countable declarations instead of a wall of failures.
//!
//! A capability the processor *has* is answered rather than declined, however inconvenient the
//! answer: the 1580 cases of `unicode-90` and `regex-classes` declare a Unicode version this
//! workspace's tables subsume, so they are adjudicated and the handful whose expectations are
//! pinned to the exact version they name fail honestly (`unicode_version_support` says what that
//! costs). Declining a set for a reason of the *driver's* own — the resources it demands — is a
//! separate decision under a label of its own, never a `dependency:` one (`costly_set`).
//!
//! Most `<result>` assertions (`assert`, `assert-eq`, …) are XPath-shaped and reuse
//! [`crate::assertions::check`], with `$result` bound to the transformation's result-tree root.
//! Seven are not — `assert-xml` (canonicalized-fragment comparison), `assert-string-value`
//! (whose XSLT-catalogue semantics normalizes whitespace unconditionally, unlike QT3's),
//! `serialization-matches` (a regex over the *serialized* output), `assert-message` (a
//! substring of an `xsl:message` log), `assert-serialization` (the serialized output compared
//! as text), `assert-serialization-error` (a serialization that must fail) and
//! `assert-result-document` (an assertion over a secondary `xsl:result-document` output) — and
//! are handled directly in `check_case`.
//!
//! # Skips are labelled
//!
//! A case this runner does not adjudicate is recorded with a *specific* reason,
//! never one collective string: a skip whose reason is `dependency:…` is a declared gap the
//! catalogue itself gates on, while any other reason is a case that would have a real verdict if
//! the runner (or the processor) could reach it. Keeping the two apart is the difference between
//! a measured conformance level and one that quietly hides its own blind spots — so the tally
//! [`crate::report::SuiteReport::skip_reasons`] carries is written to be read.
//!
//! A label is what makes a blind spot *countable*, though, not what makes it actionable: a tally
//! saying several hundred stylesheets were declined by the compiler names neither which cases nor
//! which feature. [`TRACE_VAR`] turns on a per-case disposition trace that answers both, and
//! [`SETS_VAR`] narrows a triage run to the test sets being worked on.

mod embedded;
pub mod ledger;
mod resources;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use oxixml_infoset::{AtomicValue, ExpandedName, NodeKind, NodeRef, Sequence, Tree, XdmNode};
use oxixml_xpath::XpathValue;
use oxixml_xslt::{Parameters, StaticOptions, Stylesheet, XsltError};

use resources::{
    CollectionMember, SuiteDocuments, SuiteEntities, SuiteResolver, expand_inclusions,
    parse_stylesheet_module,
};

use crate::assertions::{ErrorInfo, Verdict, check, check_with_context_item};
use crate::env;
use crate::qt3::budget::{CaseRunner, WORKER_STACK, progress, timeout_message};
use crate::report::SuiteReport;
use crate::xdm_util::{
    attribute, children_named, element_children, find_child, local_name, text_of,
};

/// Features a `<dependencies>` element may name that this processor *has*, for the sake of the
/// cases that ask for one to be **absent**.
///
/// A `<feature satisfied="false">` is not a weaker form of `satisfied="true"`: it declares that the
/// case is written for a processor *without* the named feature and describes behaviour a processor
/// with it must not show. `format-number-069b` is the clearest of them — it requires
/// `format-number()` with a decimal format that does not exist to raise an error whose code the
/// case writes as the placeholder `XXX`, which is what a processor lacking XPath 3.1 does, while
/// one that has XPath 3.1 raises `FODF1280` and is right to. Running such a case here measures a
/// processor this is not.
///
/// Kept as an explicit *supported* list rather than as the complement of [`UNSUPPORTED_FEATURES`],
/// because the two are not complements: the catalogue names features
/// ([`XML_1.1`](https://www.w3.org/TR/xml11/), for one) that appear in neither list, and treating
/// an unlisted feature as present would decline `catalog-005b`/`catalog-006b`, which this processor
/// runs and passes.
///
/// `higher_order_functions` and `namespace_axis` are here because the engine claims both (see
/// [`UNSUPPORTED_FEATURES`]), not because a case needs them to be: no `<feature>` element in the
/// catalogue writes `satisfied="false"` for either, so neither entry moves a count today. They
/// state the same fact as the property table for the shape that would otherwise be read the wrong
/// way round — a case written for a processor *without* higher-order functions is one this
/// processor should decline, exactly as it declines `format-number-069b`.
const SUPPORTED_FEATURES: &[&str] = &[
    "disabling_output_escaping",
    "XPath_3.1",
    "higher_order_functions",
    "namespace_axis",
];

/// The HTML version this processor's `html` and `xhtml` output methods serialize as by default —
/// the answer a `<default_html_version>` dependency asks for.
///
/// Serialization 3.1 [§7.3](https://www.w3.org/TR/xslt-xquery-serialization-31/#xhtml-output)
/// rules out a character in the `#x7F`–`#x9F` range for HTML 4 (`SERE0014`) and admits it for
/// HTML 5, and the catalogue pairs the two expectations over one stylesheet: `output-0195` and
/// `output-0195a` both serialize `output-0195.xsl`'s `<doc>&#x9f;</doc>` under a bare
/// `<xsl:output method="html"/>`, the first requiring the error and the second requiring the
/// character to survive. This processor writes the character, so a case gated on a default of 4
/// describes a different processor.
const DEFAULT_HTML_VERSION: &str = "5";

/// Features a `<dependencies>` element may name that this processor does not have.
///
/// An entry here is a claim about the processor, so it has to agree with the answer the processor
/// itself gives to `system-property()` — the same question, asked from inside a stylesheet, which
/// the catalogue's own cases ask and assert on. Claiming a gap the engine denies having is not a
/// cosmetic inconsistency: every case that depends on the feature is skipped as a *declared* gap,
/// which is the one skip label that says "nothing to see here", while the engine would have
/// adjudicated it. The cases that interrogate the property are gated on it too, so the
/// contradiction hides itself.
///
/// `higher_order_functions` and `namespace_axis` are deliberately absent for that reason —
/// `xsl:supports-higher-order-functions` and `xsl:supports-namespace-axis` both answer `yes`, the
/// QT3 drivers beside this one declare both, and the cases they used to gate are adjudicated
/// rather than skipped. Every name that remains is one the engine's table agrees with
/// (`xsl:supports-streaming`, `xsl:supports-dynamic-evaluation`, `xsl:is-schema-aware` and
/// `xsl:supports-backwards-compatibility` all answer `no`). The agreement is not left to a
/// comment: for every name the property table answers `yes` or `no` for, a unit test compiles a
/// stylesheet, reads the property out of the engine, and requires membership of this list to be
/// its negation. Whichever side moves first, the other has to move with it.
///
/// `XSD_1.1` is the one entry with no such property — `xsl:xsd-version` reports a version rather
/// than a capability — and it stays because the *behaviour* is XSD 1.0: `type-available-0151` and
/// `error-0905a`, which assert respectively that `xs:dateTimeStamp` is unavailable and that a
/// namespace URI outside XSD 1.0's `xs:anyURI` lexical space raises `XTDE0905`, both pass.
const UNSUPPORTED_FEATURES: &[&str] = &[
    "streaming",
    "streaming-fallback",
    "schema_aware",
    "schema-aware",
    "dynamic_evaluation",
    "XSD_1.1",
    "xpath-1.0-compatibility",
    "backwards_compatibility",
    "remote_http",
    "fn-transform-XSLT",
    "fn-transform-XSLT30",
    "fn-load-xquery-module",
    "json",
];

/// The environment variable that turns on a per-case disposition trace on standard error.
///
/// The skip tally groups by *label*, which is what makes an unadjudicated case countable; it is
/// not enough to act on one. `not-implemented:compile` says the compiler declined a stylesheet
/// and nothing about which feature it declined or which cases are waiting on it, and the engine's
/// own message — the only place that answer exists — is discarded at the point the label is
/// chosen. Setting this writes one tab-separated `id`, `kind`, `detail` line per case, where
/// `kind` is `PASS`, `FAIL`, `SKIP`, or `GAP` for the engine's message behind a
/// `not-implemented:` skip. Cross-referencing the `GAP` and `SKIP` lines for an id turns the
/// tally into a per-feature backlog with the cases attached.
///
/// Triage only, in the sense `OXIXML_QT3_PROGRESS` is: the trace observes the run and changes
/// nothing it measures.
pub const TRACE_VAR: &str = "OXIXML_XSLT30_TRACE";

/// Writes one disposition line for `id` to standard error, when [`TRACE_VAR`] is set.
///
/// `detail` is flattened onto the one line so that the trace stays greppable and line-oriented
/// however many lines an engine message or a failure's produced/expected pair runs to.
fn trace(id: &str, kind: &str, detail: &str) {
    if std::env::var_os(TRACE_VAR).is_some() {
        eprintln!("{}", trace_line(id, kind, detail));
    }
}

/// The one line [`trace`] writes, with every character that would break the record apart replaced
/// by a space.
fn trace_line(id: &str, kind: &str, detail: &str) -> String {
    let flattened: String = detail
        .chars()
        .map(|character| match character {
            '\n' | '\r' | '\t' => ' ',
            other => other,
        })
        .collect();
    format!("{id}\t{kind}\t{flattened}")
}

/// Runs the whole catalogue on a dedicated thread with a large stack — a stylesheet in the suite
/// may recurse deeply on purpose.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Ok(handle) = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_suite)
    else {
        let mut report = SuiteReport::new("xslt30");
        report.record_skip("could not spawn the harness thread");
        return vec![report];
    };
    vec![handle.join().unwrap_or_else(|_| {
        let mut report = SuiteReport::new("xslt30");
        report.record_fail("harness", "the harness thread panicked");
        report
    })]
}

fn run_suite() -> SuiteReport {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return SuiteReport::unavailable(
            "xslt30",
            match env::xml_suites_dir() {
                None => env::missing(env::XML_SUITES_DIR_VAR),
                Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, "xslt30-test/catalog.xml"),
            },
        );
    };
    let catalog = root.join("catalog.xml");
    let mut report = SuiteReport::new("xslt30");
    let Ok(text) = std::fs::read_to_string(&catalog) else {
        report.record_skip(format!("no catalog at {}", catalog.display()));
        return report;
    };
    let Ok(tree) = Tree::parse_str(&text) else {
        report.record_skip("the catalog did not parse");
        return report;
    };
    let Some(catalog_element) = tree.document_element() else {
        report.record_skip("catalogue:no-document-element");
        return report;
    };

    let selected = selected_sets();
    // Shared with the QT3 drivers (see `crate::qt3::budget`'s own documentation): every case —
    // stylesheet compilation, transformation *and* assertion checking — runs on a reused worker
    // thread under a wall-clock budget, so a stylesheet that does not terminate (or one merely
    // starved of CPU by another suite's abandoned QT3 workers when this runs as part of
    // `all_suites()`/`oxixml-conformance-report` — see that binary's own history) bounds the
    // whole catalogue's run time instead of leaving it open-ended. `progress` gives an
    // `OXIXML_QT3_PROGRESS`-gated line per test set, so a long run can be told from a stuck one
    // the same way the QT3 drivers already can.
    let mut runner = CaseRunner::new();
    for entry in element_children(&catalog_element) {
        if local_name(&entry) != "test-set" {
            continue;
        }
        // A `<test-set>` the catalogue cannot be read from is not a set with no cases: it is a set
        // whose cases nobody counted. Each of the three ways that can happen is recorded, so the
        // difference shows up in the tally rather than in a total that is quietly short.
        let Some(name) = attribute(&entry, "name") else {
            report.record_skip("catalogue:test-set-without-a-name");
            continue;
        };
        // A narrowed run records what it left out, for the same reason the driver no longer has an
        // allowlist: a `total` that is a subset of the catalogue must say so in the report, not
        // only in the shell history of whoever produced it.
        if let Some(selected) = &selected
            && !selected.iter().any(|wanted| wanted == &name)
        {
            report.record_skip("catalogue:not-selected");
            continue;
        }
        let Some(file) = attribute(&entry, "file") else {
            report.record_skip(format!("catalogue:test-set-without-a-file:{name}"));
            continue;
        };
        run_test_set(
            &root.join(&file),
            &name,
            &mut report,
            &mut runner,
            selected.is_some(),
        );
    }
    report
}

/// The environment variable that narrows a run to a comma- or whitespace-separated list of test
/// set names.
///
/// Triage only, and unlike [`TRACE_VAR`] it *does* change what the run measures: a conformance
/// number must always come from a run that leaves it unset, which is why the full catalogue is the
/// unconfigured behaviour and this is the deviation. It exists because working one mechanism —
/// `base-uri`, say — through to a fix means running its test set a dozen times, and a full
/// catalogue run is minutes each time.
///
/// So that a narrowed run can never be mistaken for a full one, every set it passes over is
/// recorded as a `catalogue:not-selected` skip: the report says how much of the catalogue was left
/// out, in the same tally as every other reason a case went unadjudicated.
pub const SETS_VAR: &str = "OXIXML_XSLT30_SETS";

/// The test sets [`SETS_VAR`] names, or `None` for the whole catalogue.
fn selected_sets() -> Option<Vec<String>> {
    parse_set_selection(&std::env::var(SETS_VAR).ok()?)
}

/// The test set names a [`SETS_VAR`] setting holds, or `None` for the whole catalogue.
///
/// An empty (or whitespace-only) setting is *not* a request for an empty run — it is how a shell
/// spells "unset" by accident — so it answers `None` and the full catalogue runs.
fn parse_set_selection(setting: &str) -> Option<Vec<String>> {
    let names: Vec<String> = setting
        .split([',', ' ', '\t', '\n'])
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();
    (!names.is_empty()).then_some(names)
}

/// What running one case produced.
///
/// The difference from [`Verdict`] is the skip: this runner records *why* it did not adjudicate a
/// case, because "skipped" alone conflates a gap the catalogue declares (a `<dependencies>` entry
/// this processor does not satisfy) with one it does not — an assertion kind the runner cannot
/// check, a stylesheet that would not compile, a structural shape the case does not have. Only the
/// first kind is accounted for; every other reason names a case whose real verdict is unknown, and
/// a reason string is what makes those countable instead of invisible.
///
/// The reason is written `category:detail` (`dependency:streaming`,
/// `unsupported-assertion:assert-warning`, …) so that a tally groups usefully when read left to
/// right, with a bare category for the reasons that have no detail to give.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    /// The case's assertions held.
    Pass,
    /// The case ran and its assertions did not hold.
    Fail(String),
    /// The case was not adjudicated, for the named reason.
    Skip(String),
}

impl Outcome {
    /// Lifts a [`Verdict`] from the shared assertion checker, labelling its skip with `reason` —
    /// the checker has no vocabulary for *why* it could not decide, only that it could not.
    fn from_verdict(verdict: Verdict, reason: impl Into<String>) -> Self {
        match verdict {
            Verdict::Pass => Self::Pass,
            Verdict::Fail(message) => Self::Fail(message),
            Verdict::Skip => Self::Skip(reason.into()),
        }
    }
}

/// One `<param>` a case supplies, from either the `<environment>` or the `<test>`.
///
/// `static` splits the two channels XSLT 3.0 keeps apart: a static parameter
/// ([§9.6](https://www.w3.org/TR/xslt-30/#static-params)) is a parameter of the *compilation*, so
/// its value must be in hand before the stylesheet is compiled and reaches the processor through
/// [`StaticOptions`]; an ordinary one is supplied per transformation through [`Parameters`].
#[derive(Debug, Clone)]
struct CaseParameter {
    name: String,
    select: String,
    is_static: bool,
}

/// A source document, the stylesheet, and the parameters a test case starts with.
#[derive(Debug, Default, Clone)]
struct Environment {
    source: Option<String>,
    /// The URI the principal source document is retrieved from, when the catalogue names it as a
    /// file rather than writing it inline.
    ///
    /// A source document parsed without one has no base URI, and the properties XDM derives from
    /// a base URI are then unavailable however well the rest of the tree is built: `fn:base-uri`,
    /// `fn:document-uri` and — the reason this exists — the `[unparsed entities]` property, whose
    /// URIs are the `SYSTEM` identifiers of the DTD's `NDATA` declarations *resolved against the
    /// document's own URI* (XDM 3.1 [§6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode)).
    /// The whole `unparsed-entity-uri` test set asserts that resolution.
    source_uri: Option<String>,
    /// The `select` a `<source role=".">` may carry, naming the **initial context node** inside
    /// the document rather than the document node itself.
    ///
    /// XSLT 3.0 [§2.3.2](https://www.w3.org/TR/xslt-30/#initiating-transformation) makes the
    /// global context item an arbitrary item, not necessarily a document node, and four catalogue
    /// entries exercise that: `mode-1105` starts at `/doc` so that a rule matching `doc` in the
    /// initial mode is the first to fire, and `strip-space-023` starts at a whitespace text node
    /// that `xsl:strip-space` then removes. Starting at the document node instead runs a
    /// different transformation.
    source_select: Option<String>,
    params: Vec<CaseParameter>,
    /// The principal stylesheet named on the `<environment>` itself. A `<test-case>` may leave its
    /// `<test>` without a `<stylesheet>` and let the environment it shares with its siblings name
    /// one (`regex-070a` and the whole `regex070` family), which is a reference by the same rules
    /// as an inline one — so the runner falls back to it rather than declining the case.
    ///
    /// Only a *principal* module counts: a `role="secondary"` `<stylesheet>` names a module the
    /// principal one imports, and compiling it as the entry point would run a different
    /// stylesheet than the case describes.
    stylesheet: Option<String>,
    /// A `<source defines-stylesheet="true">` names a source document that *contains* the
    /// stylesheet (XSLT 3.0 [§3.11](https://www.w3.org/TR/xslt-30/#embedded)), reached through an
    /// `xml-stylesheet` processing instruction rather than compiled from a file of its own.
    defines_stylesheet: bool,
    /// The documents the environment registers besides the principal source: a
    /// `<source file="…" uri="…"/>` binds a file to the URI a stylesheet retrieves it by, which
    /// need not be its name on disk.
    documents: BTreeMap<String, PathBuf>,
    /// The collections the environment declares, by the URI `fn:collection` asks for them under.
    /// The empty string is the default collection — the one a no-argument `fn:collection()`
    /// wants — and a declaration with no members is an *empty* collection rather than an absent
    /// one (see [`resources::SuiteResolver`]).
    collections: BTreeMap<String, Vec<CollectionMember>>,
}

impl Environment {
    fn parse(element: &NodeRef, directory: &Path) -> Self {
        let mut environment = Self::default();
        for child in element_children(element) {
            match local_name(&child).as_str() {
                "source" => {
                    if attribute(&child, "role").as_deref() != Some(".") {
                        // Not the principal source: a document the case can *retrieve*, under
                        // the URI the catalogue binds it to (`collection-e01` registers five,
                        // three of which its stylesheets reach through `fn:doc`).
                        if let Some(file) = attribute(&child, "file") {
                            let uri = attribute(&child, "uri").unwrap_or_else(|| file.clone());
                            environment.documents.insert(uri, directory.join(file));
                        }
                        continue;
                    }
                    if attribute(&child, "defines-stylesheet").as_deref() == Some("true") {
                        environment.defines_stylesheet = true;
                    }
                    if let Some(select) = attribute(&child, "select") {
                        environment.source_select = Some(select);
                    }
                    match attribute(&child, "file") {
                        Some(file) => {
                            let path = directory.join(file);
                            if let Some(text) = read_xml_file(&path) {
                                let uri = file_uri(&path);
                                // `xinclude="true"` asks for the document the transformation
                                // sees to be the *merged* one (see
                                // [`resources::expand_inclusions`]). Processing that cannot be
                                // carried out leaves the document exactly as it was read, so
                                // the case is decided against a document that is visibly
                                // missing its inclusions rather than disappearing behind a
                                // skip label.
                                environment.source = Some(
                                    if attribute(&child, "xinclude").as_deref() == Some("true") {
                                        expand_inclusions(&text, &uri).unwrap_or(text)
                                    } else {
                                        text
                                    },
                                );
                                environment.source_uri = Some(uri);
                            }
                        }
                        None => {
                            if let Some(content) = find_child(&child, "content") {
                                environment.source = Some(text_of(&content));
                            }
                        }
                    }
                }
                "param" => {
                    if let Some(parameter) = read_parameter(&child) {
                        environment.params.push(parameter);
                    }
                }
                "collection" => {
                    let uri = attribute(&child, "uri").unwrap_or_default();
                    let members = children_named(&child, "source")
                        .filter_map(|source| attribute(&source, "file"))
                        .map(|file| CollectionMember::new(&file, directory))
                        .collect();
                    environment.collections.insert(uri, members);
                }
                "stylesheet" => {
                    if is_principal(&child)
                        && let Some(file) = attribute(&child, "file")
                    {
                        environment.stylesheet = Some(file);
                    }
                }
                _ => {}
            }
        }
        environment
    }
}

/// The `<param>` `element` declares, or `None` when it names no value to bind.
fn read_parameter(element: &NodeRef) -> Option<CaseParameter> {
    let name = attribute(element, "name")?;
    let select = attribute(element, "select")?;
    Some(CaseParameter {
        name,
        select,
        is_static: attribute(element, "static").as_deref() == Some("yes"),
    })
}

/// Whether a `<stylesheet>`/`<package>` element names the *principal* module: the catalogue writes
/// `role="principal"` explicitly on some and leaves it off on most, and only `role="secondary"`
/// (a module the principal one imports or includes) is not one.
fn is_principal(element: &NodeRef) -> bool {
    !matches!(attribute(element, "role").as_deref(), Some("secondary"))
}

/// Runs one test set. `listed` is the name the catalogue entry gave it, used only to say which set
/// was lost when the file behind it cannot be read — the set's own `@name` governs everywhere else.
fn run_test_set(
    path: &Path,
    listed: &str,
    report: &mut SuiteReport,
    runner: &mut CaseRunner<Outcome>,
    named: bool,
) {
    let Some(text) = read_xml_file(path) else {
        report.record_skip(format!("catalogue:unreadable-test-set:{listed}"));
        return;
    };
    let Ok(tree) = Tree::parse_str(&text) else {
        report.record_skip(format!("catalogue:test-set-did-not-parse:{listed}"));
        return;
    };
    let Some(set) = tree.document_element() else {
        report.record_skip(format!("catalogue:empty-test-set:{listed}"));
        return;
    };
    let directory = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let set_name = attribute(&set, "name").unwrap_or_default();
    progress("xslt30", &set_name);

    // A `<dependencies>` on the `<test-set>` governs every case in it: the catalogue schema states
    // that in as many words, and the suite's reference runner evaluates the union
    // `$test/(.|..)/dependencies` — in document order, so the set's declaration is answered before
    // the case's own. Answering it once here also means a set the processor is declared out of
    // scope for costs nothing to enumerate: none of its cases reaches the worker thread.
    let set_gap = find_child(&set, "dependencies")
        .as_ref()
        .and_then(unsupported_dependency)
        .map(|gap| format!("dependency:{gap}"))
        // A set a whole-catalogue run declines for a cost of its own — see [`costly_set`]. A run
        // that *names* the set gets it: the decline is about what one run over 14 601 cases can
        // afford, not about the cases being unrunnable, and the measurement behind it has to stay
        // reproducible with `OXIXML_XSLT30_SETS`.
        .or_else(|| (!named).then(|| costly_set(&set_name)).flatten());

    let mut environments: std::collections::BTreeMap<String, Environment> =
        std::collections::BTreeMap::new();
    for child in element_children(&set) {
        if local_name(&child) == "environment"
            && let Some(name) = attribute(&child, "name")
        {
            environments.insert(name, Environment::parse(&child, &directory));
        }
    }
    let environments = Arc::new(environments);

    for child in element_children(&set) {
        if local_name(&child) != "test-case" {
            continue;
        }
        let case = attribute(&child, "name").unwrap_or_default();
        let id = format!("{set_name}/{case}");
        if let Some(reason) = &set_gap {
            trace(&id, "SKIP", reason);
            report.record_skip(reason.clone());
            continue;
        }
        let hint = stylesheet_hint(&child);
        // The whole case — stylesheet compilation, transformation *and* assertion checking —
        // runs on the runner's worker thread under a wall-clock budget; nothing but the verdict
        // crosses back. See `run_suite`'s own documentation for why.
        //
        // The transformation is given the *same* instant as its own deadline, taken here rather
        // than on the worker so that both clocks start together — a stylesheet slow to compile
        // would otherwise be handed a deadline that begins after the runner has already given up
        // on it. The two bounds do different jobs and both are needed: the runner's is a hard
        // wall-clock kill that always fires, the engine's is cooperative and stops the
        // computation *itself*, so a worker the runner abandons stops competing for the machine
        // instead of running to the end of the process (see `CaseRunner::abandoned`). It cannot
        // change a verdict, because it is never earlier than the budget the runner enforces.
        let deadline = std::time::Instant::now() + runner.budget();
        let child = child.clone();
        let directory = directory.clone();
        let environments = Arc::clone(&environments);
        let traced = id.clone();
        let job = Box::new(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_case(&child, &traced, &directory, &environments, deadline)
            }))
            .unwrap_or_else(|_| Outcome::Fail("the processor panicked".to_owned()))
        });
        // Read before the run: afterwards the count may include this very case, which was not
        // competing with itself.
        let contention = runner.still_running();
        let abandoned = runner.abandoned();
        let Some(outcome) = runner.run(job) else {
            let message = timeout_message(runner.budget(), contention, abandoned, &hint);
            trace(&id, "FAIL", &message);
            report.record_fail(&id, message);
            continue;
        };
        match outcome {
            Outcome::Pass => {
                trace(&id, "PASS", "");
                report.record_pass();
            }
            Outcome::Skip(reason) => {
                trace(&id, "SKIP", &reason);
                report.record_skip(reason);
            }
            Outcome::Fail(reason) => {
                trace(&id, "FAIL", &reason);
                report.record_fail(&id, reason);
            }
        }
    }
}

/// The reason a whole-catalogue run declines a test set for a cost of its own, rather than for
/// anything the catalogue declares about it; `None` for every set but one.
///
/// The reason is deliberately **not** a `dependency:` label. A dependency label says the catalogue
/// itself put the case out of this processor's audience; this one says the opposite — the cases
/// are in audience, the processor answers them, and the driver declined to spend what answering
/// them costs. Keeping the two apart is the same distinction the module documentation draws
/// between a declared gap and a blind spot, and it is why the count stays visible in the tally
/// instead of being hidden among the catalogue's own declarations.
///
/// **`unicode-90`, 1460 cases.** 1444 of them run one stylesheet, `unicode90-Gen-001.xsl`, whose
/// `<xsl:param name="source" select="doc(concat('docs/unicode-', $charclass, '.xml'))"/>` loads a
/// document holding one `<c c="…"/>` element per character of the named Unicode category, and
/// whose modes then evaluate `*/c/matches(@c, $regex)` — one regular-expression match **per
/// character**. Those documents are 121 MB over 41 files, and `docs/unicode-C.xml` alone holds
/// 984 014 elements; the 38 cases of a category each parse and walk their category's document
/// again. Measured here on the shipping tree at the shipping budget: **39 cases adjudicated in
/// 14 minutes** (13 of them timing out at 30 s), the process holding **8.7 GB** of resident
/// memory by then — which extrapolates to about **nine hours** for the set, against roughly two
/// minutes for the other 13 141 cases of the catalogue put together.
///
/// The cost is the catalogue's own data volume, not a defect this workspace can measure its way
/// out of, and it is not the abandoned-worker cascade either: a cooperative deadline is now handed
/// to every transformation (see [`run_test_set`]), and it cannot reach this set because the time
/// goes inside single uninterruptible calls — retrieving and parsing a 53 MB document, then one
/// `fn:matches` per element. The set is therefore skipped under its own name, with the cases
/// counted, and any run that wants it can have it: `OXIXML_XSLT30_SETS=unicode-90` runs it in
/// full.
fn costly_set(set_name: &str) -> Option<String> {
    (set_name == "unicode-90").then(|| format!("cost:{set_name}"))
}

/// A short description of `case` for a timeout message: the `<test>`'s `<stylesheet>` (or
/// `<package>`) file it compiles, which — unlike a QT3 expression or query — is short and almost
/// always present, so it is a more useful "what was running" hint than the whole `<test-case>`
/// subtree would be.
fn stylesheet_hint(case: &NodeRef) -> String {
    find_child(case, "test")
        .as_ref()
        .and_then(find_child_stylesheet_file)
        .unwrap_or_else(|| "(no <stylesheet> file)".to_owned())
}

fn find_child_stylesheet_file(test: &NodeRef) -> Option<String> {
    element_children(test)
        .iter()
        .find(|child| matches!(local_name(child).as_str(), "stylesheet" | "package"))
        .and_then(|child| attribute(child, "file"))
}

/// Runs one `<test-case>`. `id` names it for [`TRACE_VAR`] only — the verdict itself carries the
/// id back through [`SuiteReport`], and nothing here decides anything by which case it is.
///
/// `deadline` is the instant the caller's own wall-clock budget expires, handed to the
/// transformation so that a case the runner abandons stops computing rather than running on for
/// the life of the process.
fn run_case(
    case: &NodeRef,
    id: &str,
    directory: &Path,
    environments: &std::collections::BTreeMap<String, Environment>,
    deadline: std::time::Instant,
) -> Outcome {
    // The same deadline, for everything the case does *outside* the transformation: a `use-when`
    // evaluated while the stylesheet is compiled, a catalogue selector, and the XPath-shaped
    // assertions afterwards. `Parameters::with_deadline` below covers the transformation itself,
    // which the engine may run on a thread of its own — a thread-local declaration cannot reach
    // it, and this one cannot reach a stylesheet the compiler is still reading. Together they
    // bound the whole case.
    let _limits = oxixml_xpath::EvaluationLimits::declare_deadline(deadline);
    if let Some(dependencies) = find_child(case, "dependencies")
        && let Some(gap) = unsupported_dependency(&dependencies)
    {
        return Outcome::Skip(format!("dependency:{gap}"));
    }
    let Some(test) = find_child(case, "test") else {
        return Outcome::Skip("no-test".to_owned());
    };
    let Some(result) = find_child(case, "result") else {
        return Outcome::Skip("no-result".to_owned());
    };

    let environment = match find_child(case, "environment") {
        Some(element) => match attribute(&element, "ref") {
            Some(reference) => environments.get(&reference).cloned().unwrap_or_default(),
            None => Environment::parse(&element, directory),
        },
        None => Environment::default(),
    };

    let module = match principal_stylesheet(&test, &environment, directory) {
        Ok(module) => module,
        Err(reason) => return Outcome::Skip(reason),
    };

    // Every `<param>` the case supplies, from the environment it references and from its own
    // `<test>`, in that order. Reading them here — before the stylesheet is compiled — is what a
    // *static* parameter needs: XSLT 3.0 §9.6 makes it a parameter of the compilation, so a
    // shadow attribute or `use-when` that reads one has to see its value while the stylesheet
    // tree is still being expanded.
    let params: Vec<CaseParameter> = environment
        .params
        .iter()
        .cloned()
        .chain(element_children(&test).iter().filter_map(read_parameter))
        .collect();

    let resolver = Arc::new(SuiteResolver::new(
        directory,
        environment.documents.clone(),
        environment.collections.clone(),
    ));

    // A catalogue entry that supplies no `<source role=".">` gets no global context item at
    // all — XSLT 3.0 §2.3.4 makes that an explicit, legal invocation shape, not the same thing
    // as an empty document. A fabricated `<empty/>` document would be observably wrong: a named
    // template reached through `<initial-template>` may declare `<xsl:context-item as="…"
    // use="optional"/>`, which must see an *absent* focus here, not a real (and wrongly typed)
    // element node.
    //
    // The source is read before the stylesheet is compiled because an embedded module *is* a
    // node of this tree, and the same tree is still the transformation's input.
    let source_tree = match &environment.source {
        Some(text) => {
            let mut entities = SuiteEntities::new(directory);
            match Tree::parse_str_with_resolver(text, source_options(&environment), &mut entities) {
                Ok(tree) => Some(tree),
                Err(_) => return Outcome::Skip("source-did-not-parse".to_owned()),
            }
        }
        None => None,
    };

    // What the compiler is handed, and the base URI it compiles under. The base URI is the
    // module's own document URI either way — for an embedded module that is the *source
    // document's*, which is what an `xsl:import` written inside it resolves against.
    let (input, base_uri) = match &module {
        Module::File(path) => {
            let Some(text) = read_xml_file(path) else {
                return Outcome::Skip("unreadable-stylesheet".to_owned());
            };
            // A module this runner cannot parse is handed on as *text*, so that the compiler's
            // own parse reports the error the catalogue grades (`XTSE0165`) — the one answer an
            // application cannot construct on the engine's behalf.
            let input = match parse_stylesheet_module(&text, path, directory) {
                Some(tree) => ModuleInput::Tree(tree.root()),
                None => ModuleInput::Text(text),
            };
            (input, Some(file_uri(path)))
        }
        Module::Embedded => {
            let Some(tree) = &source_tree else {
                return Outcome::Skip("embedded-stylesheet:no-source".to_owned());
            };
            let Some(element) = embedded::embedded_module(&tree.root()) else {
                return Outcome::Skip("embedded-stylesheet:not-named".to_owned());
            };
            (ModuleInput::Tree(element), environment.source_uri.clone())
        }
    };

    // `UNSUPPORTED_FEATURES` declares `backwards_compatibility` a gap, so the processor is
    // driven in the matching mode: an element whose effective version is 1.0 is `XTDE0160`
    // rather than silently evaluated under XSLT 3.0 semantics. Without this the harness would
    // claim one thing to the catalogue's dependency filter and do another.
    let mut options = StaticOptions::new()
        .with_resolver(resolver)
        .with_backwards_compatibility(false);
    if let Some(uri) = base_uri {
        options = options.with_base_uri(uri);
    }
    for parameter in params.iter().filter(|parameter| parameter.is_static) {
        options = options.with_static_parameter(
            parameter.name.clone(),
            selector_value(&parameter.select, None, directory),
        );
    }

    let outcome = match &input {
        ModuleInput::Tree(root) => Stylesheet::compile_with(root, options),
        ModuleInput::Text(text) => Stylesheet::compile_str_with(text, options),
    };
    let compiled = match outcome {
        Ok(compiled) => compiled,
        // `OXIXML0001` is this crate's own "not implemented" code: the stylesheet uses a
        // feature that is a documented gap, so the case is out of scope rather than failed. The
        // engine's message names *which* gap, and the skip label deliberately does not — so it is
        // traced here, where it still exists.
        Err(error) if error.is("OXIXML0001") => {
            trace(id, "GAP", &error.to_string());
            return Outcome::Skip("not-implemented:compile".to_owned());
        }
        Err(error) => return check_error_outcome(&result, &error, directory),
    };

    // This runs on `CaseRunner`'s worker thread, which — like `run`'s own 256 MiB thread — is
    // sized generously (512 MiB; see `crate::qt3::budget::WORKER_STACK`) precisely because suite
    // stylesheets recurse deeply on purpose (call-template-1003 walks a thousand elements by
    // recursion), so the engine's default instantiation ceiling — sized for an ordinary 8 MiB
    // main-thread stack — is raised to match. At the measured cost of ~7 KiB of native stack per
    // level, 10 000 levels is about 70 MiB: far more than any case in the suite needs, and
    // comfortably inside either thread's stack. The step limit still bounds a genuinely runaway
    // stylesheet, and the case runner's wall-clock budget bounds one that does not even loop.
    let mut parameters = Parameters::new()
        .with_step_limit(4_000_000)
        .with_recursion_limit(10_000)
        .with_stack_size(WORKER_STACK)
        .with_deadline(deadline);
    // The base output URI: "an implementation defined base output URI as chosen by the test
    // driver" (`admin/catalog-schema.xsd`), which the catalogue nevertheless *names* for the
    // cases that ask what it is — `<output file="results/current-output-uri-002.xml"/>`. It is
    // where the principal result would be written, so `fn:current-output-uri` reports it and an
    // `xsl:result-document/@href` resolves against it.
    if let Some(uri) = base_output_uri(&test, directory) {
        parameters = parameters.with_base_output_uri(uri);
    }
    // A static parameter is supplied through *both* channels: its compiled value governs (the
    // compiler binds it before the invocation's), and a stylesheet may also declare an ordinary
    // `xsl:param` of the same name that the transformation still has to see.
    for parameter in &params {
        parameters = parameters.with_parameter(
            parameter.name.clone(),
            selector_value(&parameter.select, None, directory),
        );
    }
    // A `<param>` child of `<initial-template>` is not a stylesheet parameter: XSLT 3.0
    // [§2.3.4](https://www.w3.org/TR/xslt-30/#invoking-initial-template) supplies the initial
    // template's *own* `xsl:param` values through a separate channel, which is why
    // `initial-template-003` can bind `a` to 77 as a stylesheet parameter and to 1234 as the
    // entry point's, and expect both to be visible. `tunnel="yes"` picks the tunnel channel.
    if let Some(initial) = find_child(&test, "initial-template")
        && let Some(name) = attribute(&initial, "name")
    {
        parameters = parameters
            .with_initial_template(expanded(&initial, &name))
            .with_initial_template_parameters(entry_point_parameters(&initial, directory));
    }
    // XSLT 3.0 [§2.3.5](https://www.w3.org/TR/xslt-30/#invoking-initial-function)'s third entry
    // point: the transformation *is* a call of a named stylesheet function, whose arguments the
    // catalogue supplies positionally. The arity of that argument list is what selects the
    // function, so a name and arity no stylesheet declares is `XTDE0041` — which is the whole of
    // `error-0041a`, and the only reason this entry point is reachable from an allowed test set.
    if let Some(initial) = find_child(&test, "initial-function")
        && let Some(name) = attribute(&initial, "name")
    {
        parameters = parameters
            .with_initial_function(expanded(&initial, &name))
            .with_initial_function_arguments(entry_point_arguments(&initial, directory));
    }
    // The second entry point (XSLT 3.0 [§2.3.3](https://www.w3.org/TR/xslt-30/#invoking-initial-mode)):
    // template rules of a named mode are applied to the initial match selection. Its `<param>`
    // children are the *initial template rule's* own parameters, the same channel
    // `<initial-template>` uses above and not the stylesheet parameters — `initial-mode-004` binds
    // `a` and a tunnelled `my:b` that the matching `<xsl:template mode="flobble">` declares as its
    // own `xsl:param`s.
    let initial_mode = find_child(&test, "initial-mode");
    if let Some(initial) = &initial_mode
        && let Some(name) = attribute(initial, "name")
    {
        parameters = parameters
            .with_initial_mode(expanded(initial, &name))
            .with_initial_mode_parameters(entry_point_parameters(initial, directory));
    }

    let document_root = source_tree.as_ref().map(Tree::root);
    // The global context item the transformation starts from: the document node, or the one node
    // inside it a `<source select="…">` names. See [`Environment::source_select`].
    let source_root = match (&document_root, &environment.source_select) {
        (Some(root), Some(select)) => Some(initial_context_node(select, root, directory)),
        // A `<source role=".">` with a `select` and no `file` names no document to look inside:
        // the expression *is* the source, and the node it constructs is the global context item.
        (None, Some(select)) => constructed_context_node(select, directory),
        (root, _) => root.clone(),
    };

    // `<initial-mode select="…">` supplies the *initial match selection* (XSLT 3.0 §2.3.3): the
    // sequence the initial mode's template rules are applied to, which the catalogue writes as an
    // XPath expression over the source document rather than as a document of its own. It is
    // evaluated here, once the source tree exists, because `mode-1802` selects from it — against
    // the document node, which is what such an expression is written relative to.
    if let Some(initial) = &initial_mode
        && let Some(select) = attribute(initial, "select")
    {
        parameters = parameters.with_initial_match_selection(selector_value(
            &select,
            document_root.as_ref(),
            directory,
        ));
    }

    let raw_result = match raw_result_wanted(&test) {
        Ok(wanted) => wanted,
        Err(reason) => return Outcome::Skip(reason),
    };

    match compiled.transform_optional(source_root.as_ref(), &parameters) {
        Ok(transformation) => {
            let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::from(if raw_result {
                transformation.as_sequence()
            } else {
                Sequence::from(transformation.root())
            }));
            // `<result>` is a wrapper around the one assertion (often an `all-of`/`any-of` tree)
            // the case makes, exactly as on the error path: `check_case` decides *assertions*,
            // and handing it the wrapper would decide every case "unsupported assertion kind".
            let Some(assertion) = element_children(&result).into_iter().next() else {
                return Outcome::Skip("empty-result".to_owned());
            };
            let produced = Produced {
                serialized: transformation.as_str(),
                messages: transformation.messages(),
                secondary: transformation.secondary_results(),
            };
            check_case(&assertion, Some(produced), &outcome, directory)
        }
        Err(error) if error.is("OXIXML0001") => {
            trace(id, "GAP", &error.to_string());
            Outcome::Skip("not-implemented:transform".to_owned())
        }
        Err(error) => check_error_outcome(&result, &error, directory),
    }
}

/// What a *successful* transformation produced, for the assertion kinds that look past the result
/// tree: the serialized principal result, the `xsl:message` log, and the secondary
/// `xsl:result-document` outputs as `(href, serialized)` pairs.
#[derive(Clone, Copy)]
struct Produced<'a> {
    serialized: &'a str,
    messages: &'a [String],
    secondary: &'a [(String, String)],
}

/// Checks a `<result>` whose transformation failed, converting the error into the shared
/// checker's normalized shape.
fn check_error_outcome(result: &NodeRef, error: &XsltError, directory: &Path) -> Outcome {
    let outcome: Result<XpathValue, ErrorInfo> = Err(error_info(error));
    let Some(assertion) = element_children(result).into_iter().next() else {
        return Outcome::Skip("empty-result".to_owned());
    };
    check_case(&assertion, None, &outcome, directory)
}

/// Checks one `<result>` assertion tree, recursing through `all-of`/`any-of`/`not` itself so that
/// the XSLT-specific leaf kinds (which need what the transformation [`Produced`] and, for
/// `assert-xml`, a directory to resolve an expected-output file from) can sit alongside the ones
/// the shared XPath-shaped checker already handles.
///
/// A leaf this runner cannot decide answers [`Outcome::Skip`] with a reason naming the kind, and a
/// connective propagates the reason of whichever child could not be decided — so a case skipped
/// because one leaf of an `all-of` was undecidable says *which* leaf, not merely that the tree as
/// a whole was not reached.
fn check_case(
    assertion: &NodeRef,
    success: Option<Produced<'_>>,
    outcome: &Result<XpathValue, ErrorInfo>,
    directory: &Path,
) -> Outcome {
    let kind = local_name(assertion);
    match kind.as_str() {
        "all-of" => {
            let mut skipped = None;
            for child in element_children(assertion) {
                match check_case(&child, success, outcome, directory) {
                    Outcome::Pass => {}
                    Outcome::Fail(reason) => return Outcome::Fail(reason),
                    Outcome::Skip(reason) => skipped = skipped.or(Some(reason)),
                }
            }
            match skipped {
                Some(reason) => Outcome::Skip(reason),
                None => Outcome::Pass,
            }
        }
        "any-of" => {
            let mut last_failure = None;
            let mut skipped = None;
            for child in element_children(assertion) {
                match check_case(&child, success, outcome, directory) {
                    Outcome::Pass => return Outcome::Pass,
                    Outcome::Fail(reason) => last_failure = Some(reason),
                    Outcome::Skip(reason) => skipped = skipped.or(Some(reason)),
                }
            }
            match (last_failure, skipped) {
                (Some(reason), _) => Outcome::Fail(reason),
                (None, Some(reason)) => Outcome::Skip(reason),
                (None, None) => Outcome::Skip("empty-assertion:any-of".to_owned()),
            }
        }
        "not" => match element_children(assertion).into_iter().next() {
            Some(child) => match check_case(&child, success, outcome, directory) {
                Outcome::Pass => Outcome::Fail("the negated assertion held".to_owned()),
                Outcome::Fail(_) => Outcome::Pass,
                Outcome::Skip(reason) => Outcome::Skip(reason),
            },
            None => Outcome::Skip("empty-assertion:not".to_owned()),
        },
        // `assert-xml` is an assertion about the *result tree*, not about what the stylesheet's
        // `xsl:output` makes of it: the catalogue defines it against "a serialization of the
        // expression result using the default serialization parameters `method="xml"`
        // `indent="no"` `omit-xml-declaration="yes"`" (`admin/catalog-schema.xsd`), and holds when
        // canonicalizing that agrees with canonicalizing the expectation. Canonicalizing the tree
        // itself is that, and it is the only way to keep the serializer's own method-specific
        // rewriting out of the comparison — the `html`/`xhtml` methods insert a
        // `<meta http-equiv="Content-Type">` (Serialization 3.1
        // [§11.2](https://www.w3.org/TR/xslt-xquery-serialization-31/#xhtml-output)) that the
        // result tree does not contain, so `bug-1901` would otherwise be judged against a
        // document the transformation never built. A result the case asked for as a raw sequence
        // has no tree to walk, and keeps the serialized comparison.
        "assert-xml" => match success {
            None => Outcome::Skip("no-result-tree:assert-xml".to_owned()),
            Some(produced) => {
                let Some(expected_text) = expected_text(assertion, directory) else {
                    return Outcome::Skip("missing-expected-file:assert-xml".to_owned());
                };
                let actual = match result_document(success, outcome) {
                    Some(root) => canonical_tree(root),
                    None => canonical(produced.serialized),
                };
                let wanted = canonical(&expected_text);
                if actual == wanted {
                    Outcome::Pass
                } else if canonical_fragment(strip_declaration(&expected_text)).is_none() {
                    // The expectation is not a fragment this parser accepts, so `canonical` gave
                    // back its collapsed text instead of a canonical form and the two sides were
                    // never comparable. A mismatch here says nothing about the processor, so it is
                    // not recorded as though it did.
                    Outcome::Skip("unparseable-expectation:assert-xml".to_owned())
                } else {
                    Outcome::Fail(format!("expected `{wanted}`, produced `{actual}`"))
                }
            }
        },
        "assert-string-value" => match (success, outcome) {
            (Some(_), Ok(value)) => {
                let Some(actual) = crate::assertions::joined_string_value(value) else {
                    return Outcome::Skip("not-a-string-value:assert-string-value".to_owned());
                };
                let expected = text_of(assertion);
                // `runner/assert.xsl` normalizes whitespace unconditionally for assertions in
                // the XSLT catalogue namespace, whatever `normalize-space` on the element says.
                let (actual, expected) = (
                    crate::assertions::collapse(&actual),
                    crate::assertions::collapse(&expected),
                );
                if actual == expected {
                    Outcome::Pass
                } else {
                    Outcome::Fail(format!("string value `{actual}`, expected `{expected}`"))
                }
            }
            _ => Outcome::Skip("no-result-tree:assert-string-value".to_owned()),
        },
        "serialization-matches" => match success {
            Some(produced) => {
                let pattern = text_of(assertion);
                let flags = attribute(assertion, "flags").unwrap_or_default();
                match oxixml_regex::Regex::xpath(&pattern, &flags) {
                    Ok(regex) if regex.is_match(produced.serialized) => Outcome::Pass,
                    Ok(_) => Outcome::Fail(format!(
                        "`{}` does not match `{pattern}`",
                        produced.serialized
                    )),
                    Err(_) => Outcome::Skip("unsupported-regex:serialization-matches".to_owned()),
                }
            }
            None => Outcome::Skip("no-result-tree:serialization-matches".to_owned()),
        },
        // `assert-serialization` compares the *serialized* principal result as text, which is
        // what makes it different from `assert-xml`: indentation, escaping, the XML declaration
        // and the output method all matter here, and canonicalizing either side would throw away
        // exactly what the case is testing. The two texts are compared after normalizing line
        // endings and trimming the outer whitespace, which is framing the catalogue's own XML
        // indentation (or a fixture file's trailing newline) contributes and no serializer
        // controls.
        "assert-serialization" => match success {
            None => Outcome::Skip("no-result-tree:assert-serialization".to_owned()),
            Some(produced) => {
                let Some(expected) = expected_text(assertion, directory) else {
                    return Outcome::Skip("missing-expected-file:assert-serialization".to_owned());
                };
                let (actual, expected) = (
                    serialization_text(produced.serialized),
                    serialization_text(&expected),
                );
                if actual == expected {
                    Outcome::Pass
                } else {
                    Outcome::Fail(format!("serialized `{actual}`, expected `{expected}`"))
                }
            }
        },
        // `assert-serialization-error` asserts that the transformation itself runs cleanly and
        // the *serializer* then rejects the result tree. A processor that detects the same fault
        // earlier — while compiling the `xsl:output` declaration that causes it, say — has still
        // raised the error the case names, so an error outcome is graded by its code exactly as
        // `<error>` is; only a run that produced serialized output without complaint is a failure.
        "assert-serialization-error" => match success {
            None => Outcome::from_verdict(
                crate::assertions::check_error(assertion, outcome),
                "unsupported-assertion:assert-serialization-error".to_owned(),
            ),
            Some(produced) => Outcome::Fail(format!(
                "expected the serialization error {}, but the result serialized to `{}`",
                attribute(assertion, "code").unwrap_or_else(|| "*".to_owned()),
                serialization_text(produced.serialized)
            )),
        },
        // `assert-result-document` asserts that a *secondary* result tree was written to the named
        // URI and satisfies the assertion nested inside it. That inner assertion is decided
        // against the secondary document in its own right: its serialization for a text-shaped
        // leaf, its parsed form as the focus for an XPath-shaped one.
        "assert-result-document" => match success {
            None => Outcome::Skip("no-result-tree:assert-result-document".to_owned()),
            Some(produced) => {
                let Some(uri) = attribute(assertion, "uri") else {
                    return Outcome::Skip("no-uri:assert-result-document".to_owned());
                };
                let Some(inner) = element_children(assertion).into_iter().next() else {
                    return Outcome::Skip("empty-assertion:assert-result-document".to_owned());
                };
                let Some((href, serialized)) = produced
                    .secondary
                    .iter()
                    .find(|(href, _)| names_result_document(href, &uri))
                else {
                    return Outcome::Fail(format!(
                        "no result document was written to `{uri}` (written: {})",
                        written_uris(produced.secondary)
                    ));
                };
                // Reparsing the serialization loses where the document was written *to*, and
                // `result-document-0102` asserts exactly that: `base-uri(/)` must end in the
                // secondary document's own href, and an `xml:base` inside it resolves against
                // that. The href the engine recorded is the one the catalogue means, so it is
                // handed back to the parser as the document URI.
                let options = oxixml_infoset::ParseOptions::default().document_uri(href.clone());
                let tree = Tree::parse_str_with(serialized, options).ok();
                let outcome: Result<XpathValue, ErrorInfo> = Ok(match &tree {
                    Some(tree) => XpathValue::from(Sequence::from(tree.root())),
                    // A secondary result serialized as text or JSON is not a document; a
                    // text-shaped leaf still decides it, an XPath-shaped one sees no focus.
                    None => XpathValue::default(),
                });
                let secondary = Produced {
                    serialized,
                    messages: produced.messages,
                    secondary: &[],
                };
                check_case(&inner, Some(secondary), &outcome, directory)
            }
        },
        "assert-message" => match success {
            Some(produced) => {
                let expected = text_of(assertion);
                if produced
                    .messages
                    .iter()
                    .any(|message| message.contains(&expected))
                {
                    Outcome::Pass
                } else {
                    // The catalogue cannot say *which* `xsl:message` a case's text should appear
                    // in, and this processor's message log is not yet complete enough for an
                    // absent one to be evidence of a defect, so this is left undecided rather
                    // than failed — under its own reason, so the count stays visible.
                    Outcome::Skip("message-not-produced:assert-message".to_owned())
                }
            }
            None => Outcome::Skip("no-result-tree:assert-message".to_owned()),
        },
        "error" if success.is_none() => Outcome::from_verdict(
            check(assertion, outcome),
            "unsupported-assertion:error".to_owned(),
        ),
        "error" => Outcome::Fail(format!(
            "expected the error {}, but the transformation succeeded",
            attribute(assertion, "code").unwrap_or_default()
        )),
        // Unlike QT3's, this catalogue's XPath-shaped assertions are evaluated with the result
        // document node as the context item (`admin/catalog-schema.xsd`, `assert`), which is what
        // lets `function-0701` write `count(//td) = 64` and `call-template-1003` write
        // `/out = "…"` without naming `$result` at all.
        _ => Outcome::from_verdict(
            check_with_context_item(assertion, outcome, result_document(success, outcome)),
            format!("unsupported-assertion:{kind}"),
        ),
    }
}

/// The text an assertion compares against: the contents of the file its `file` attribute names,
/// resolved relative to the test-set directory, or — with no such attribute — the element's own
/// text. `None` only when a named file could not be read.
fn expected_text(assertion: &NodeRef, directory: &Path) -> Option<String> {
    match attribute(assertion, "file") {
        Some(file) => std::fs::read_to_string(directory.join(file)).ok(),
        None => Some(text_of(assertion)),
    }
}

/// A serialized result, prepared for the byte-for-byte comparison `assert-serialization` makes:
/// line endings normalized to `\n` and the outer whitespace removed.
///
/// Nothing else is normalized. The two differences that *are* removed are ones no serializer
/// controls: a fixture file's line endings are whatever checked it out, and an inline expectation
/// is written inside an indented XML element, so it always carries the surrounding newlines and
/// spaces of the catalogue's own layout.
fn serialization_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_owned()
}

/// Whether a written result document's `href` is the one an `assert-result-document` names.
///
/// The catalogue writes the URI relative to "an implementation defined base output URI as chosen
/// by the test driver" (`admin/catalog-schema.xsd`), while the engine records the `href` already
/// resolved against the stylesheet's own base URI — so the two are compared by the reference the
/// catalogue actually controls: the whole URI, or its final path segments.
fn names_result_document(href: &str, uri: &str) -> bool {
    href == uri || href.ends_with(&format!("/{uri}"))
}

/// The URIs a transformation actually wrote to, for the message an unmatched
/// `assert-result-document` reports; `"none"` when it wrote no secondary result at all.
fn written_uris(secondary: &[(String, String)]) -> String {
    if secondary.is_empty() {
        return "none".to_owned();
    }
    secondary
        .iter()
        .map(|(href, _)| href.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The result document node an XPath-shaped assertion is evaluated against: the single node the
/// successful outcome carries, which is the transformation's own result tree root.
///
/// `None` when the transformation raised an error instead, so a `<result>` reached through
/// [`check_error_outcome`] keeps evaluating against no focus — there is no result document for
/// one to be.
fn result_document<'a>(
    success: Option<Produced<'_>>,
    outcome: &'a Result<XpathValue, ErrorInfo>,
) -> Option<&'a NodeRef> {
    success?;
    match outcome {
        Ok(value) => match value.iter().as_slice() {
            [item] => item.as_node(),
            _ => None,
        },
        Err(_) => None,
    }
}

/// The [`ErrorInfo`] an [`XsltError`] normalizes to, with the code written the way the catalogue
/// writes an `error/@code` — see [`ErrorInfo::qualified`].
fn error_info(error: &XsltError) -> ErrorInfo {
    ErrorInfo::qualified(
        error.namespace(),
        error.code().unwrap_or_default(),
        error.to_string(),
    )
}

/// The first dependency `dependencies` declares that this processor does not satisfy, named so
/// that the skip it causes says which gap it was; `None` when every one of them holds.
fn unsupported_dependency(dependencies: &NodeRef) -> Option<String> {
    for child in element_children(dependencies) {
        let value = attribute(&child, "value").unwrap_or_default();
        let satisfied = attribute(&child, "satisfied")
            .map(|value| value == "true")
            .unwrap_or(true);
        let kind = local_name(&child);
        match kind.as_str() {
            "feature" if satisfied && UNSUPPORTED_FEATURES.contains(&value.as_str()) => {
                return Some(value);
            }
            // The other direction: a case written for a processor *without* a feature this one
            // has. See [`SUPPORTED_FEATURES`] — it asserts behaviour the feature's absence
            // produces, so a processor that has the feature cannot satisfy it and is not meant to.
            "feature" if !satisfied && SUPPORTED_FEATURES.contains(&value.as_str()) => {
                return Some(format!("feature-present:{value}"));
            }
            // Which HTML version the *default* `html`/`xhtml` serialization is, which the
            // catalogue gates in both directions over one stylesheet. See
            // [`DEFAULT_HTML_VERSION`].
            "default_html_version" if (value.trim() == DEFAULT_HTML_VERSION) != satisfied => {
                return Some(format!("{kind}:{value}"));
            }
            // A case restricted to a version before 3.0 asserts behaviour 3.0 deliberately
            // changed (e.g. `match="$v"` is a static error pre-3.0, legal in 3.0); this
            // processor implements 3.0 semantics, so such a case is out of scope, not a failure.
            "spec"
                if !value.trim().is_empty()
                    && value
                        .split_whitespace()
                        .all(|token| token == "XSLT10" || token == "XSLT20") =>
            {
                return Some(format!("spec:{value}"));
            }
            // `year_component_values` asks which corner of the year value space the processor
            // covers, and the catalogue gates *both* directions on the answer: `satisfied="false"`
            // marks a case only a processor **without** the capability can pass, `satisfied="true"`
            // one only a processor with it can. The two are written as pairs over the same
            // stylesheet — `date-094a` requires `xs:date('0000-09-05')` to raise an error while
            // `date-094n` requires `xs:date('0000-02-29')` to succeed — so leaving the dependency
            // unanswered runs both halves of every pair and counts one of them a failure whatever
            // the processor does.
            "year_component_values" if year_component_support(&value) != Some(satisfied) => {
                return Some(format!("{kind}:{value}"));
            }
            // Streamability analysis, asked as a capability of its own rather than as the
            // `streaming` feature: a `<posture-and-sweep>` case gives the processor an XPath
            // expression and a context posture and asks what posture and sweep XSLT 3.0
            // [§19.8](https://www.w3.org/TR/xslt-30/#streamability-analysis) assigns the result.
            // This processor performs no streamability analysis at all — `xsl:supports-streaming`
            // answers `no` — so it has no answer to give, in either direction.
            "sweep_and_posture" => return Some(format!("{kind}:{value}")),
            // Which Unicode version the processor's character tables carry. Unlike the
            // capabilities below this one is *answered*, because the question it asks has an
            // answer: a case declaring `unicode-version value="9.0"` needs a processor whose
            // character properties are at least Unicode 9.0's, and this workspace's tables are
            // newer than every version the catalogue asks about. See
            // [`unicode_version_support`] for what that costs — the two sets it admits contain
            // cases whose expectations are pinned to the *exact* version they name, and those
            // fail rather than being hidden behind a label.
            "unicode-version" if unicode_version_support(&value) != Some(satisfied) => {
                return Some(format!("{kind}:{value}"));
            }
            // Capabilities pinned to a *version* or to a named locale resource. This processor
            // implements one version of each and does not offer to behave as another, so a case
            // that asks for a specific one is out of scope whichever way it asks — the value is
            // carried into the label so that the tally says which version was asked for rather
            // than merely that a version was.
            "xsd-version"
            | "unicode-normalization-form"
            | "default-language"
            | "language"
            | "calendar"
            | "formatIntegerSequence" => {
                return Some(match value.trim() {
                    "" => kind,
                    version => format!("{kind}:{version}"),
                });
            }
            _ => {}
        }
    }
    None
}

/// Whether this processor has the year-value-space capability a `year_component_values`
/// dependency names, and `None` for a capability the catalogue names that this list does not
/// know — an unrecognized one is undecidable, so the case that declares it is skipped either way.
///
/// All three the `xslt30-test` catalogue writes are answered yes. `oxixml_xsd` implements the
/// proleptic Gregorian calendar of [XSD 1.1 Part 2
/// §D.3.2](https://www.w3.org/TR/xmlschema11-2/#dateTime-lexical-mapping), whose `yearFrag` is
/// four digits *minimum* with no upper bound and admits both `0000` and a leading minus sign, and
/// no layer above it narrows that: year zero, negative years and years above 9999 alike parse,
/// compare and serialize. The only ceiling is the width of the internal timeline, which reports
/// `FODT0001` for a year of nineteen-odd digits — orders of magnitude past the four this
/// dependency asks about, so it does not qualify the answer.
fn year_component_support(value: &str) -> Option<bool> {
    match value.trim() {
        "support year zero" | "support negative year" | "support year above 9999" => Some(true),
        _ => None,
    }
}

/// Whether this processor's character tables are at least as new as the Unicode version a
/// `unicode-version` dependency names, and `None` for a version this list does not know — an
/// unrecognized one is undecidable, so the case that declares it is skipped either way.
///
/// The catalogue asks about two: `unicode-90` (1460 cases) declares `9.0` and `regex-classes`
/// (120) declares `6.0`. `oxixml_unicode`'s tables are Unicode 17.0.0, which subsumes both in the
/// sense the dependency asks about — a case needing the digits Unicode 9.0 added finds them — so
/// both are answered yes and the cases are adjudicated.
///
/// What that does *not* buy is agreement on data the later versions changed, and the two sets
/// contain cases that depend on exactly that: a `regex-classes` case comparing the members of
/// `\p{Cf}` against a reference document written for Unicode 6.0 cannot hold on 17.0, because the
/// category gained members in between. Those cases fail, and failing is the point — a skip label
/// would say "out of scope" about a case this processor runs to completion and answers, differing
/// from the expectation only in character data whose version the expectation itself fixes. The
/// alternative measured before this was admission: 1580 cases behind one blanket label, whose
/// passing majority (the great bulk of both sets) was invisible.
fn unicode_version_support(value: &str) -> Option<bool> {
    match value.trim() {
        "6.0" | "9.0" => Some(true),
        _ => None,
    }
}

/// Reads an XML file the catalogue names, honouring the encoding it announces.
///
/// A dozen stylesheets and a handful of source documents in `xslt30-test` are ISO-8859-1 (and
/// declare it), so reading them as UTF-8 fails outright — which used to make a stylesheet
/// `unreadable-stylesheet` and, worse, silently leave a source document unsupplied. The encoding
/// is settled the way [XML 1.0 Appendix F](https://www.w3.org/TR/xml/#sec-guessing) settles it:
/// byte-order mark and prologue pattern first, then the declaration's own `encoding`
/// pseudo-attribute. The same reading the XSD and QT3 runners in this crate already use.
fn read_xml_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let sniffed = oxixml_encoding::sniff_xml_encoding(&bytes);
    let mut text = String::with_capacity(bytes.len());
    let mut decoder = sniffed.best_effort_encoding().new_decoder();
    decoder
        .decode_to_string(bytes.get(sniffed.bom_len..)?, &mut text, true)
        .ok()?;
    Some(text)
}

/// The URI a file the catalogue names is retrieved from, spelled the one way this runner spells
/// one: `file://` and the path as the platform writes it, neither percent-encoded nor
/// canonicalized. Every base URI the runner hands the processor — a stylesheet's, a source
/// document's — is built here, so a relative reference resolved from one against the other stays
/// inside the suite directory it was written for.
fn file_uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

/// The options the principal source document is parsed with: its own document URI, when the
/// catalogue named a file to read it from, and the external subset of its document type
/// declaration.
///
/// A document parsed without a URI is a document with no base URI, which is a different document
/// from the one the catalogue describes — see [`Environment::source_uri`]. A document whose
/// `<!DOCTYPE … SYSTEM "…">` is not read is missing whatever that subset declares: the
/// `unparsed-entity-uri` set's `-50` puts its `NDATA` entities there, and `<!ATTLIST>` defaults
/// declared outside the internal subset are part of the same document.
fn source_options(environment: &Environment) -> oxixml_infoset::ParseOptions {
    let options = oxixml_infoset::ParseOptions::default().read_external_subset(true);
    match &environment.source_uri {
        Some(uri) => options.document_uri(uri.clone()),
        None => options,
    }
}

/// The principal stylesheet module the case compiles, or the reason there is none to compile.
///
/// The `<test>`'s own `<stylesheet>` is the usual answer; a `role="secondary"` one names a module
/// the principal module imports, so it is passed over rather than compiled as an entry point. A
/// `<test>` that names no stylesheet at all falls back to the one its `<environment>` names — a
/// whole family of cases shares one environment and states the stylesheet there once
/// (`regex-070a` and its siblings).
///
/// A `<package>` entry point is compiled the same way. A package *is* a stylesheet module — XSLT
/// 3.0 [§3.5](https://www.w3.org/TR/xslt-30/#packages) makes `xsl:package` an alternative document
/// element to `xsl:stylesheet`, and the catalogue writes `<package role="principal" file="…"/>`
/// for a case whose entry point happens to be one. Handing it to the compiler is what makes the
/// engine's own answer visible: a case that needs packaging is then declined by name
/// (`OXIXML0001`, traced as a `GAP` line saying which construct), not by the harness refusing to
/// look.
///
/// The one remaining shape is not a file at all: a `defines-stylesheet` source holds the module
/// *inside* the source document, which [`Module::Embedded`] names and [`embedded`] finds.
fn principal_stylesheet(
    test: &NodeRef,
    environment: &Environment,
    directory: &Path,
) -> Result<Module, String> {
    let children = element_children(test);
    if let Some(element) = children.iter().find(|child| {
        matches!(local_name(child).as_str(), "stylesheet" | "package") && is_principal(child)
    }) {
        return match attribute(element, "file") {
            Some(file) => Ok(Module::File(directory.join(file))),
            None => Err(format!("no-{}-file", local_name(element))),
        };
    }
    if let Some(file) = &environment.stylesheet {
        return Ok(Module::File(directory.join(file)));
    }
    if environment.defines_stylesheet {
        return Ok(Module::Embedded);
    }
    Err("no-stylesheet".to_owned())
}

/// Where the principal stylesheet module of a case is: a file of its own, or an element of the
/// principal source document.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Module {
    /// A stylesheet module in a file beside the test set — the usual case.
    File(PathBuf),
    /// A module embedded in the principal source document, named by an `xml-stylesheet`
    /// processing instruction (XSLT 3.0 [§3.11](https://www.w3.org/TR/xslt-30/#embedded)).
    Embedded,
}

/// The principal module in the form the compiler is given it.
///
/// A tree is the usual and the more faithful of the two — it is the only form an embedded module
/// has, and the only one whose entity references this runner has already resolved. Text is what
/// remains when the module is not well-formed: the compiler's own parse then produces the error,
/// under the code the catalogue grades it by.
enum ModuleInput {
    /// A module this runner parsed, rooted at the element to compile.
    Tree(NodeRef),
    /// A module left for the compiler to parse.
    Text(String),
}

/// The value a catalogue `select` attribute denotes: the XPath expression it holds, evaluated with
/// `context` as the context item.
///
/// The catalogue's `<param>` and `<initial-mode>` selectors are XPath, not literals — `true()`,
/// `xs:date('2019-01-01')` and `doc('mode-14.xml')//v[position() = 1 to 5]` all appear — so they
/// are compiled and evaluated. An expression this engine cannot compile falls back to
/// [`literal_value`], which recognizes the quoted strings and integers that make up the bulk of
/// them, so a gap in the XPath engine cannot make a case unsupplied.
fn selector_value(select: &str, context: Option<&NodeRef>, directory: &Path) -> XpathValue {
    match evaluate_selector(select, context, directory) {
        Some(value) => value,
        None => XpathValue::from(literal_value(select)),
    }
}

/// Compiles and evaluates one catalogue selector, answering `None` when either step fails.
fn evaluate_selector(
    select: &str,
    context: Option<&NodeRef>,
    directory: &Path,
) -> Option<XpathValue> {
    let compiled = oxixml_xpath::XPath::compile(
        select,
        oxixml_xpath::XPathVersion::XPath31,
        oxixml_xpath::StaticContext::new(),
    )
    .ok()?;
    let mut dynamic = compiled
        .context()
        .with_document_resolver(Arc::new(SuiteDocuments {
            directory: directory.to_path_buf(),
        }));
    if let Some(item) = context {
        dynamic = dynamic.with_context_item(item.clone());
    }
    compiled.evaluate_value_in(&dynamic).ok()
}

/// The node a `<source role="." select="…">` names inside `document`, or `document` itself when
/// the expression does not denote exactly one node.
///
/// The fallback is deliberate rather than a decline: an expression this engine cannot compile, or
/// one denoting something that is not a single node, leaves the case running from the document
/// node it would have run from anyway — a gap in the XPath engine cannot turn a case into an
/// unsupplied source, exactly as [`selector_value`] refuses to leave a parameter unbound.
fn initial_context_node(select: &str, document: &NodeRef, directory: &Path) -> NodeRef {
    let selected = evaluate_selector(select, Some(document), directory);
    let node = selected.and_then(|value| match value.iter().as_slice() {
        [item] => item.as_node().cloned(),
        _ => None,
    });
    node.unwrap_or_else(|| document.clone())
}

/// The node a `<source role="." select="…"/>` *builds*, for the environment that names no file to
/// look inside — `id-043` writes `select="parse-xml('&lt;root/>')"`, whose value is a document the
/// expression constructs rather than one the catalogue ships.
///
/// Evaluated with no context item, because there is nothing for the expression to be relative to.
/// `None` when it denotes anything other than exactly one node, which leaves the transformation
/// with no global context item at all — the same state a `<source role=".">`-less environment
/// produces, and an honest one: a source that could not be built is not a source of some other
/// shape.
fn constructed_context_node(select: &str, directory: &Path) -> Option<NodeRef> {
    match evaluate_selector(select, None, directory)?
        .iter()
        .as_slice()
    {
        [item] => item.as_node().cloned(),
        _ => None,
    }
}

/// Turns the `select` of an environment parameter into a value, handling the literals the suite
/// actually uses.
fn literal_value(select: &str) -> AtomicValue {
    let trimmed = select.trim();
    if let Some(inner) = trimmed
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
    {
        return AtomicValue::string(inner);
    }
    if let Some(inner) = trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    {
        return AtomicValue::string(inner);
    }
    match trimmed.parse::<i64>() {
        Ok(value) => AtomicValue::integer(value),
        Err(_) => AtomicValue::string(trimmed),
    }
}

/// The `{uri}local` form of an `EQName` the catalogue writes on `<initial-mode>`/
/// `<initial-template>`, resolving the prefix against the namespaces in scope on the element
/// itself.
fn expanded(element: &NodeRef, lexical: &str) -> String {
    let trimmed = lexical.trim();
    let Some((prefix, local)) = trimmed.split_once(':') else {
        return trimmed.to_owned();
    };
    if trimmed.starts_with('{') || trimmed.starts_with("Q{") {
        return trimmed.to_owned();
    }
    let mut current = Some(element.clone());
    while let Some(node) = current {
        for binding in node.namespaces() {
            let bound = binding
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default();
            if bound == prefix {
                return format!("{{{}}}{local}", binding.string_value());
            }
        }
        current = node.parent();
    }
    trimmed.to_owned()
}

/// The [`ExpandedName`] a catalogue `name` attribute denotes, resolved the same way
/// [`expanded`] resolves it and then split back into its two parts.
fn expanded_name(element: &NodeRef, lexical: &str) -> ExpandedName {
    let resolved = expanded(element, lexical);
    let braced = resolved
        .strip_prefix("Q{")
        .or_else(|| resolved.strip_prefix('{'));
    match braced.and_then(|rest| rest.split_once('}')) {
        Some((namespace, local)) => ExpandedName::new(Some(namespace), local),
        None => ExpandedName::new(None::<&str>, resolved.as_str()),
    }
}

/// The parameters a catalogue `<initial-template>` or `<initial-mode>` supplies to the entry point
/// it names, as the `(name, value, tunnel)` triples
/// [`Parameters::with_initial_template_parameters`] and
/// [`Parameters::with_initial_mode_parameters`] take.
///
/// These are the entry point's own `xsl:param` values (XSLT 3.0 §2.3.3/§2.3.4), which is a
/// different channel from the stylesheet parameters a `<test>`-level `<param>` supplies — the two
/// may bind the same name to different values in one case, and both bindings have to be visible
/// where they apply. A `<param>` with no `select` binds nothing, exactly as [`read_parameter`]
/// treats one at the outer level.
fn entry_point_parameters(
    entry: &NodeRef,
    directory: &Path,
) -> Vec<(ExpandedName, XpathValue, bool)> {
    children_named(entry, "param")
        .filter_map(|parameter| {
            let name = attribute(&parameter, "name")?;
            let select = attribute(&parameter, "select")?;
            Some((
                expanded_name(&parameter, &name),
                selector_value(&select, None, directory),
                attribute(&parameter, "tunnel").as_deref() == Some("yes"),
            ))
        })
        .collect()
}

/// The positional arguments a catalogue `<initial-function>` calls its entry point with, in the
/// order the `<param>` children are written.
///
/// Unlike the entry-point parameters of [`entry_point_parameters`], these are *arguments*: they
/// have no names of their own (the schema's `unNamedParameters` group), and how many there are
/// decides which arity of the named function is called. A `<param>` with no `select` names no
/// value, exactly as [`read_parameter`] treats one at the outer level.
fn entry_point_arguments(entry: &NodeRef, directory: &Path) -> Vec<XpathValue> {
    children_named(entry, "param")
        .filter_map(|parameter| attribute(&parameter, "select"))
        .map(|select| selector_value(&select, None, directory))
        .collect()
}

/// The base output URI a `<test>` names, resolved against the test set's own directory, or `None`
/// when the case declares that the transformation has none.
///
/// XSLT 3.0 [§2.3.1](https://www.w3.org/TR/xslt-30/#invoking-initial-mode) leaves the base output
/// URI to the invocation, and the catalogue supplies it as `<output file="results/…"/>`: the file
/// the principal result would be written to, which is what `fn:current-output-uri` answers and
/// what an `xsl:result-document/@href` is resolved against.
///
/// One spelling says the opposite — that there is *no* base output URI: `#absent`, the catalogue's
/// own sentinel, under which `current-output-uri-013` and `-015` assert
/// `empty(fn:current-output-uri())`. It is answered by supplying none, so the engine reports the
/// absence rather than a URI the case denied it.
///
/// An empty `file=""` is *not* that. It names the test set's own directory — an empty relative
/// reference resolves to the base it is resolved against — and `current-output-uri-014` asserts
/// exactly that reading, requiring the base output URI to contain the test set's path.
fn base_output_uri(test: &NodeRef, directory: &Path) -> Option<String> {
    let file = attribute(&find_child(test, "output")?, "file")?;
    let trimmed = file.trim();
    if trimmed == "#absent" {
        return None;
    }
    Some(file_uri(&directory.join(trimmed)))
}

/// Whether the case asks for the transformation's *raw result sequence* rather than a result
/// tree, or the reason this runner cannot present it the way the case asks.
///
/// `tree="no"` is the whole of it: the catalogue schema makes the attribute the counterpart of
/// `xsl:output/@build-tree`, so the transformation's result *is* the raw sequence and no tree is
/// built from it. `initial-template-004` returns eight `xs:decimal`s and asserts `deep-equal`
/// against them, which a result tree would have flattened into one text node long before the
/// assertion sees it; `initial-function-100b` supplies one `xs:integer` and asserts its *type*,
/// which a tree would have turned into a text node just as thoroughly.
///
/// `result-var` names a variable the assertion may reference, and does not decide whether a tree
/// is built. [`crate::assertions`] evaluates every XPath-shaped assertion with the outcome bound
/// to `$result`, which is the name the catalogue's own convention uses (and the only one it
/// writes), so a case naming any other one is left undecided rather than silently answered against
/// a name it did not ask for.
fn raw_result_wanted(test: &NodeRef) -> Result<bool, String> {
    let Some(output) = find_child(test, "output") else {
        return Ok(false);
    };
    if attribute(&output, "tree").as_deref() != Some("no") {
        return Ok(false);
    }
    match attribute(&output, "result-var").as_deref() {
        None | Some("result") => Ok(true),
        Some(other) => Err(format!("unsupported-result-var:{other}")),
    }
}

/// A canonical form of an XML fragment, for comparing an `assert-xml` expectation with the
/// serialized result: parsed and rewritten with attributes in a fixed order and without an XML
/// declaration, so two serializations differing only in attribute order or the declaration
/// compare equal. Text that does not parse falls back to a whitespace-collapsed comparison.
fn canonical(text: &str) -> String {
    let trimmed = strip_declaration(text);
    canonical_fragment(trimmed)
        .unwrap_or_else(|| trimmed.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// The canonical form of `text` as an XML fragment, or `None` when it is not one this parser
/// accepts.
///
/// The distinction matters where the text is an *expectation*: a comparison between a canonical
/// form and a fallback collapsed string is not a comparison at all, so a caller that would
/// otherwise report a mismatch as the processor's fault needs to know which of the two it holds.
/// The catalogue contains such expectations — `namespace-2603` and its siblings write
/// `<b xmlns:n=""/>`, and undeclaring a *prefixed* namespace is legal only in XML 1.1
/// ([Namespaces in XML 1.1 §5](https://www.w3.org/TR/xml-names11/#nsc-NSDeclared)), which is not
/// the version this parser reads a fragment as.
fn canonical_fragment(text: &str) -> Option<String> {
    let wrapped = format!("<canonical-wrapper>{text}</canonical-wrapper>");
    let tree = Tree::parse_str(&wrapped).ok()?;
    let root = tree.document_element()?;
    let mut out = String::new();
    for child in root.children() {
        write_canonical(&child, &mut out);
    }
    Some(out)
}

/// The same canonical form [`canonical`] produces, taken from a result tree directly rather than
/// from a serialization of it.
///
/// `root` is the transformation's result document node, whose children are the top-level nodes of
/// the result — the same shape [`canonical`] reaches by wrapping a fragment in an element and
/// walking *its* children, so the two sides of an `assert-xml` comparison are written by one
/// function and differ only where the trees do.
///
/// The outer whitespace is trimmed for the same reason [`canonical`] trims it off the expectation:
/// the catalogue writes an inline `assert-xml` body as CDATA inside an indented element, so it
/// always carries the newlines and spaces of the catalogue's own layout and never the ones the
/// stylesheet produced. Trimming only one side would decide a case on framing neither the
/// stylesheet nor the processor controls.
fn canonical_tree(root: &NodeRef) -> String {
    let mut out = String::new();
    for child in root.children() {
        write_canonical(&child, &mut out);
    }
    out.trim().to_owned()
}

/// The XML text of `text` with the two things in front of its document entity that are encoding,
/// not content, removed: a byte-order mark and the XML declaration.
///
/// Seven expectation files in the catalogue are written with a UTF-8 BOM (`date-061.out` is the
/// one an `assert-xml` reads). [XML 1.0 Appendix F](https://www.w3.org/TR/xml/#sec-guessing) makes
/// it an encoding signature rather than a character of the document, and leaving it in place puts
/// it *inside* the wrapper element [`canonical`] builds — which no longer parses, so the whole
/// comparison silently degrades to the whitespace-collapsed fallback and can never hold.
fn strip_declaration(text: &str) -> &str {
    let trimmed = text.trim_start_matches('\u{feff}').trim();
    match trimmed.strip_prefix("<?xml") {
        Some(rest) => match rest.find("?>") {
            Some(end) => trimmed[end + 7..].trim(),
            None => trimmed,
        },
        None => trimmed,
    }
}

fn write_canonical(node: &NodeRef, out: &mut String) {
    match node.node_kind() {
        NodeKind::Element => {
            let name = node
                .node_name()
                .map(|name| name.to_string())
                .unwrap_or_default();
            out.push('<');
            out.push_str(&name);
            let mut attributes: Vec<(String, String)> = node
                .attributes()
                .into_iter()
                .map(|attribute| {
                    (
                        attribute
                            .node_name()
                            .map(|name| name.to_string())
                            .unwrap_or_default(),
                        attribute.string_value(),
                    )
                })
                .collect();
            attributes.sort();
            for (name, value) in attributes {
                out.push_str(&format!(" {name}=\"{}\"", escape(&value)));
            }
            out.push('>');
            for child in node.children() {
                write_canonical(&child, out);
            }
            out.push_str(&format!("</{name}>"));
        }
        NodeKind::Text => out.push_str(&escape(&node.string_value())),
        NodeKind::Comment => out.push_str(&format!("<!--{}-->", node.string_value())),
        NodeKind::ProcessingInstruction => {
            let target = node
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default();
            out.push_str(&format!("<?{target} {}?>", node.string_value()));
        }
        _ => {}
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests;
