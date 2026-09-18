//! W3C **QT3** test suite (FOTS) runner, for both the XPath and XQuery engines.
//!
//! The two engines share one catalogue (`catalog.xml` plus one `test-set` file per `<test-set>`
//! entry) and almost the same `<environment>`/`<dependency>` vocabulary, so the catalogue
//! discovery, environment parsing and document resolution live here once; [`xpath_driver`] and
//! [`xquery_driver`] each add only what their engine's context type needs (a
//! [`oxixml_xpath::StaticContext`] vs. a [`oxixml_xquery::StaticQueryContext`], most of the
//! difference) and how a case's outcome converts to the [`crate::assertions`] checker's
//! normalized `Result<XpathValue, ErrorInfo>` shape.
//!
//! A case is *runnable* when every `<dependency>` it declares is decidable (a `spec` naming an
//! XPath/XQuery version, or a `feature` the engine implements) and its `<environment>` needs
//! nothing this driver cannot reproduce (no `<schema>`, no external module for XQuery). Anything
//! else — including every assertion kind [`crate::assertions::check`] does not implement, such
//! as `assert-xml` or `assert-serialization*` — is counted as skipped, never as a failure: this
//! runner's job is to measure what it *can* decide, honestly.
//!
//! # Document resolution
//!
//! `Documents` answers `fn:doc`/`fn:collection`/`fn:unparsed-text`/`fn:json-doc` for both
//! engines. Four things make it more than a lookup table:
//!
//! * a `<source uri="...">`/`<resource uri="...">` registers the document/text under *that*
//!   string too, not only under its `file` path — a case that reaches the resource through a
//!   `<param>`-bound URI (the common `fn:doc($uri)` shape) needs exactly this key to resolve;
//! * a URI nothing pre-registered (most often a relative path written directly into a query
//!   body, as the JSON Test Suite cases do throughout `misc-JsonTestSuite`) falls back to reading
//!   it from disk, relative to the test-set's own directory — as written, and then with its
//!   percent-escapes decoded, since a URI names a file by its *decoded* path (see
//!   `Documents::read_fallback`) — rather than reporting it unconditionally unavailable;
//! * a collection member is registered under an absolute `file:` URI, and carries that URI as its
//!   document URI, so a `fn:document-uri`/`fn:doc` round trip finds the same node back (see
//!   `file_uri`); and
//! * every byte read — pre-registered or fallback — goes through real encoding detection
//!   ([`oxixml_encoding`]) instead of an infallible-looking `read_to_string` that actually just
//!   silently drops a UTF-16 or ISO-8859-1 fixture as unavailable. See `decode_resource` for
//!   the exact byte-order-mark/`$encoding`-argument/UTF-8-default priority this follows, chosen
//!   to match what the catalogue itself grades as acceptable.

pub(crate) mod budget;
pub(crate) mod serialization;
pub mod xpath_driver;
pub mod xquery_driver;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_encoding::{Encoding, detect_bom, sniff_xml_encoding};
use oxixml_fn::numeric::DecimalFormat;
use oxixml_infoset::{ExpandedName, Item, NodeRef, ParseOptions, Tree, XdmNode};
use oxixml_xpath::{DocumentResolver, StaticContext, TextRetrievalError, XPath, XPathVersion};

use crate::env;
use crate::report::SuiteReport;
use crate::xdm_util::{attribute, children_named, element_children, local_name};

/// Locates the `qt3tests` directory (or a `qt3*`-prefixed sibling) under
/// `OXIXML_XML_SUITES_DIR`.
#[must_use]
pub fn qt3_root() -> Option<PathBuf> {
    env::xml_suite_subdir("qt3tests", "catalog.xml")
}

/// The environment variable that narrows a run to the test sets whose catalogue `file=` attribute
/// starts with one of its space-separated prefixes (`"fn/ op/"`, `"prod/ContextItemDecl.xml"`, …).
///
/// Triage only, exactly like the per-case budget override: running one test set takes a second
/// where the whole catalogue takes minutes, which is what makes a single fix's effect measurable
/// in isolation. A *reported* conformance number must never come from a narrowed run — it
/// measures a fraction of the corpus and says nothing about the rest.
///
/// That last rule is enforced rather than asked for: when this variable narrows a run, the
/// resulting [`SuiteReport`]'s own name says so — it names the narrowing — and a line goes to
/// standard error. A partial report that looked complete would be worse than no report at all.
pub const ONLY_VAR: &str = "OXIXML_QT3_ONLY";

/// The test-set prefixes [`ONLY_VAR`] selects, empty when it is unset — which means "every set".
fn requested_test_sets() -> Vec<String> {
    std::env::var(ONLY_VAR)
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

/// The name a driver's report carries: its plain name for a full run, and one that names the
/// narrowing for a run [`ONLY_VAR`] restricted — so a partial measurement identifies itself
/// wherever the report is printed, tabulated or pasted.
pub(crate) fn suite_name(driver: &str) -> String {
    let requested = requested_test_sets();
    if requested.is_empty() {
        return driver.to_owned();
    }
    let selection = requested.join(" ");
    eprintln!("[{driver}] PARTIAL RUN: {ONLY_VAR} narrows this to `{selection}`");
    format!("{driver} (partial: {ONLY_VAR}={selection})")
}

/// Whether a test-set file is one this run covers. See [`ONLY_VAR`].
fn test_set_selected(requested: &[String], file: &str) -> bool {
    requested.is_empty() || requested.iter().any(|prefix| file.starts_with(prefix))
}

/// A reason [`qt3_root`] returned `None`, for an unavailable [`SuiteReport`].
#[must_use]
pub fn unavailable_reason() -> String {
    match env::xml_suites_dir() {
        None => env::missing(env::XML_SUITES_DIR_VAR),
        Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, "qt3tests*/catalog.xml"),
    }
}

/// Runs both the XPath and XQuery drivers over the whole catalogue, on a dedicated thread with a
/// large stack.
///
/// The catalogue includes cases built specifically to stress a recursive-descent implementation
/// — deeply parenthesized/nested expressions among them — so, like [`crate::xslt30::run`], this
/// runs on its own thread with a generous stack rather than the ambient one a test harness or the
/// `oxixml-conformance-report` binary happens to have.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Ok(handle) = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_both)
    else {
        let mut report = SuiteReport::new("qt3");
        report.record_skip("could not spawn the harness thread");
        return vec![report];
    };
    handle.join().unwrap_or_else(|_| {
        let mut xpath = SuiteReport::new("qt3-xpath");
        xpath.record_fail("harness", "the harness thread panicked");
        let mut xquery = SuiteReport::new("qt3-xquery");
        xquery.record_fail("harness", "the harness thread panicked");
        vec![xpath, xquery]
    })
}

fn run_both() -> Vec<SuiteReport> {
    vec![xpath_driver::run(), xquery_driver::run()]
}

/// The failures of `report` that are **wrong answers** — the ones that say something about this
/// processor, as opposed to the ones that say something about the machine it ran on.
///
/// This is what a QT3 run can honestly be gated on. Both drivers answer every case they attempt
/// correctly, so the residue at a budget that isolates the processor is empty; but a case whose
/// cost is a large fraction of the per-case budget is measuring the machine as much as the
/// processor, and at the shipping budget on a loaded machine a handful of the heaviest cases —
/// a 200 001-iteration FLWOR building a map per iteration, a ray tracer — run out of it. Those
/// come back marked `TIMEOUT` and are excluded here, which is why the assertion
/// this supports holds on a busy machine without weakening into "the pass rate is high enough".
///
/// The number quoted for release is a separate matter and comes from a run at a budget that
/// removes the variable altogether (`OXIXML_QT3_CASE_BUDGET_SECS=180`), where there are no
/// timeouts to exclude.
#[must_use]
pub fn wrong_answers(report: &SuiteReport) -> Vec<&crate::report::CaseFailure> {
    report
        .failures
        .iter()
        .filter(|failure| !failure.message.starts_with(budget::TIMEOUT_MARKER))
        .collect()
}

/// Reads `bytes` as XML text, honouring a byte-order mark or an `encoding="..."` declaration in
/// the prologue (see [`oxixml_encoding::sniff_xml_encoding`]) rather than assuming UTF-8 — the
/// catalogue and several of its fixtures are plain UTF-8, but a handful of document fixtures
/// declare `encoding="ISO-8859-1"`, and a naive `read_to_string` would either mis-decode them or,
/// for a non-ASCII-compatible encoding, fail outright and be indistinguishable from a missing
/// file.
fn decode_xml_bytes(bytes: &[u8]) -> Option<String> {
    let sniffed = sniff_xml_encoding(bytes);
    let mut text = String::with_capacity(bytes.len());
    let mut decoder = sniffed.best_effort_encoding().new_decoder();
    let payload = bytes.get(sniffed.bom_len..)?;
    decoder.decode_to_string(payload, &mut text, true).ok()?;
    Some(text)
}

fn parse_file(path: &Path) -> Option<Tree> {
    let bytes = fs::read(path).ok()?;
    let text = decode_xml_bytes(&bytes)?;
    Tree::parse_str(&text).ok()
}

/// Parses a source document, giving it a document URI so `fn:document-uri`/`fn:base-uri` have
/// something to report.
fn parse_source(path: &Path) -> Option<Tree> {
    parse_source_as(path, path.to_string_lossy().as_ref())
}

/// The same as [`parse_source`], with an explicit document URI. A *collection member* needs one:
/// see [`file_uri`].
fn parse_source_as(path: &Path, document_uri: &str) -> Option<Tree> {
    let bytes = fs::read(path).ok()?;
    let text = decode_xml_bytes(&bytes)?;
    let options = ParseOptions::default().document_uri(document_uri.to_owned());
    Tree::parse_str_with(&text, options).ok()
}

/// The `file:` URI naming `path`, the form a collection member's document URI has to take.
///
/// `fn:document-uri` reports whatever the document was parsed under, and `fn:doc` resolves its
/// argument against the static base URI before looking it up — so a bare filesystem path, which
/// has no scheme, is a *relative* reference. `fn-collection/collection-009` is written around
/// exactly that round trip (`for $doc in $c return doc(document-uri($doc))`) and its environment
/// declares an `http:` static base URI, against which `/…/docs/bib.xml` resolves to
/// `http://www.w3.org/…/docs/bib.xml` — a document nothing has. An absolute `file:` URI resolves
/// to itself, so the round trip finds the very node the collection handed out.
fn file_uri(path: &Path) -> String {
    format!("file://{}", path.to_string_lossy())
}

/// The filesystem path a `file:` URI names, or `None` for a URI of any other shape.
///
/// Both local spellings are accepted: `file:///some/path`, which is what resolving a relative
/// reference against a `file:` base URI produces, and the authority-less `file://some/path` that
/// [`file_uri`] itself writes — after the `file://` prefix both leave a path starting at the root,
/// which is the only form that can name a file here.
fn file_uri_path(uri: &str) -> Option<&str> {
    uri.strip_prefix("file://")
        .filter(|path| path.starts_with('/'))
}

/// The absolute `file:` key a *relative* declared URI resolves to once the engine has resolved it
/// against the default static base URI, or `None` for a URI that already carries a scheme.
///
/// An `<environment>` may name a resource or source by a relative URI and let its queries use that
/// name: `app/UseCaseJSON.xml` declares `<resource file="UseCaseJSON/mildred.json"
/// uri="mildred.json"/>` and its queries then write `fn:json-doc("mildred.json")`. With a static
/// base URI in force that reference arrives at the resolver *already resolved*, as
/// `file:///…/app/mildred.json` — which names neither the declared `uri` nor the file's real path
/// under `UseCaseJSON/`. Registering the resolved form alongside the other keys is what keeps the
/// declaration reachable.
///
/// The "has a scheme" test is a bare search for a colon, deliberately the same one the engine
/// applies before resolving (`fn:doc`/`fn:unparsed-text` leave a URI containing `:` alone), so the
/// key registered here and the URI that will be looked up are produced by the same rule.
fn resolved_relative_key(directory: &Path, uri: &str) -> Option<String> {
    (!uri.contains(':')).then(|| file_uri(&directory.join(uri)))
}

/// The static base URI a case gets when its environment declares none: the `file:` URL of the
/// test-set file whose `<test>` element the query text lives in.
///
/// This is the catalogue's own convention, stated in `catalog-schema.xsd` on `static-base-uri`:
/// "In the absence of this element, or of a base URI declaration in the query prolog, the static
/// base URI of a query is the file containing the text of the query". Several cases are written
/// around it and cannot be decided without it — `prod/BaseURIDecl.xml`'s `K2-BaseURIProlog-5` is
/// `declare base-uri ""; ends-with(fn:static-base-uri(), "prod/BaseURIDecl.xml")`, which resolves
/// the empty reference against precisely this default and inspects the answer.
///
/// An environment that spells `<static-base-uri uri="#UNDEFINED"/>` is asserting that *no* base
/// URI is available and must keep none; see [`Environment::explicit_no_base_uri`].
fn apply_default_base_uri(environment: &mut Environment, test_set: &Path) {
    if environment.base_uri.is_none() && !environment.explicit_no_base_uri {
        environment.base_uri = Some(file_uri(test_set));
    }
}

/// One `<source>` child of an `<environment>`.
#[derive(Debug, Clone, Default)]
pub struct Source {
    /// `"."` selects the context item; `"$name"` additionally binds a variable to the document's
    /// root; anything else (usually absent) contributes only to `fn:doc`/`fn:collection` lookups.
    pub role: String,
    /// The path to read from disk: relative to the test-set directory for a locally-defined
    /// environment, or already absolute for a catalogue-root shared one (see `absolutize`).
    pub file: String,
    /// The declared `uri` attribute, if any: the document is also registered under this exact
    /// string, which is what a `<param select="'that URI'">`-bound `fn:doc($param)` call looks it
    /// up by.
    pub uri: Option<String>,
}

/// One `<environment>`, whichever engine reads it.
#[derive(Default, Clone)]
pub struct Environment {
    /// Source documents, in declaration order.
    pub sources: Vec<Source>,
    /// `(prefix, uri)` namespace bindings in scope for the case's expression/query.
    pub namespaces: Vec<(String, String)>,
    /// `(name, select)` — `select` is evaluated as XPath once the case's static context exists.
    pub parameters: Vec<(String, String)>,
    /// Text resources by URI, for `fn:unparsed-text`/`fn:json-doc`: `(uri, file)`. The bytes are
    /// read raw and decoded on demand (see `decode_resource`), never eagerly as UTF-8, so a
    /// UTF-16 or ISO-8859-1 fixture is not silently treated as unavailable.
    pub resources: Vec<(String, String)>,
    /// Named collections: `(collection uri, member file paths)`; the empty string is the default
    /// collection (`fn:collection()`/`fn:collection(())`).
    pub collections: Vec<(String, Vec<String>)>,
    /// Collections whose members are the *items* an expression produces rather than documents read
    /// from disk: `(collection uri, query)`, again with the empty string naming the default
    /// collection.
    ///
    /// F&O 3.1 §14.7.3 maps a collection URI to "a sequence of items", not to a sequence of
    /// documents, and the catalogue exercises exactly that: `fn/collection.xml` declares
    /// `<collection uri="…/integers"><query>1 to 10</query></collection>` and two siblings holding
    /// strings and doubles, while `app/UseCaseR31.xml` builds its collections by parsing JSON.
    pub collection_queries: Vec<(String, String)>,
    /// The `select` of a `<context-item>` child: an expression whose value is the case's context
    /// item, evaluated once the static context exists.
    ///
    /// Distinct from a `<source role=".">`, which can only ever supply a document node.
    /// `prod/ContextItemDecl.xml` needs the general form throughout — `contextDecl-020` supplies
    /// the string `'London'` to a query declaring `context item as xs:integer`, and expects the
    /// resulting type error rather than the "no context item" one a driver that ignored the
    /// element would produce.
    pub context_item_select: Option<String>,
    /// Decimal formats declared on this environment: `(name, properties)`, `None` naming the
    /// unnamed/default format `fn:format-number` uses when its own third argument is omitted.
    pub decimal_formats: Vec<(Option<ExpandedName>, DecimalFormat)>,
    /// The static base URI, from a `<static-base-uri>` child.
    pub base_uri: Option<String>,
    /// The URI of the `<collation default="true">` child, when the environment declares one.
    ///
    /// This is the *default collation* of the case's static context — what `fn:compare`,
    /// `fn:contains`, an order-by clause and every other collation-sensitive operation uses when
    /// no collation argument is given. `app/fo-spec-examples.xml` writes each of its collation
    /// examples twice, once passing the URI explicitly and once declaring it default here, so
    /// ignoring the element decides half of each pair against the codepoint collation.
    pub default_collation: Option<String>,
    /// Whether a `<static-base-uri>` child spelled the catalogue's `#UNDEFINED` sentinel: not a
    /// literal URI, but a deliberate assertion that *no* base URI is available, so a relative
    /// `fn:doc`/`fn:unparsed-text`/`fn:json-doc` reference must fail to resolve rather than be
    /// read from the test-set directory (see `Documents`'s `allow_fallback`).
    pub explicit_no_base_uri: bool,
    /// A `<schema>` child means the case needs schema awareness, which neither engine has.
    pub needs_schema: bool,
}

fn read_environment(node: &NodeRef) -> Environment {
    let mut environment = Environment::default();
    for child in element_children(node) {
        match local_name(&child).as_str() {
            "source" => {
                let role = attribute(&child, "role").unwrap_or_default();
                if let Some(file) = attribute(&child, "file") {
                    let uri = attribute(&child, "uri");
                    environment.sources.push(Source { role, file, uri });
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
                    // A `<param name="test:x" xmlns:test="…">` binds its own prefix on the `param`
                    // element rather than declaring it as a sibling `<namespace>`, which is the
                    // shape `prod/VarDecl.external.xml`'s `extvardeclwithouttype-24` and
                    // `extvardeclwithtype-24` use — so the prefix is resolved against that
                    // element's own in-scope namespaces and added to the environment's bindings,
                    // where the driver's static context picks it up. An explicit `<namespace>` for
                    // the same prefix wins: it is the environment's own declaration.
                    if let Some((prefix, _)) = name.split_once(':')
                        && let Some(uri) =
                            oxixml_fn::qnames::namespace_uri_for_prefix(Some(prefix), &child)
                        && !environment
                            .namespaces
                            .iter()
                            .any(|(bound, _)| bound == prefix)
                    {
                        environment.namespaces.push((prefix.to_owned(), uri));
                    }
                    environment.parameters.push((name, select));
                }
            }
            "context-item" => {
                if let Some(select) = attribute(&child, "select") {
                    environment.context_item_select = Some(select);
                }
            }
            "resource" => {
                if let (Some(file), Some(uri)) =
                    (attribute(&child, "file"), attribute(&child, "uri"))
                {
                    environment.resources.push((uri, file));
                }
            }
            "collection" => {
                let uri = attribute(&child, "uri").unwrap_or_default();
                let files: Vec<String> = children_named(&child, "source")
                    .filter_map(|source| attribute(&source, "file"))
                    .collect();
                if files.is_empty() {
                    // No `<source>` members: the collection is whatever its `<query>` evaluates
                    // to. Several `<query>` children would each be a separate collection under
                    // one URI, which the catalogue never writes, so the first one is the one.
                    if let Some(query) = children_named(&child, "query").next() {
                        environment
                            .collection_queries
                            .push((uri, query.string_value()));
                    }
                } else {
                    environment.collections.push((uri, files));
                }
            }
            "decimal-format" => {
                let name =
                    attribute(&child, "name").map(|lexical| resolve_local_qname(&lexical, &child));
                environment
                    .decimal_formats
                    .push((name, read_decimal_format(&child)));
            }
            // A `<collation>` without `default="true"` only declares that the URI is one the
            // processor should recognize, which every driver here already does through the
            // function library's own collation registry; only the default one changes the static
            // context.
            "collation" => {
                if attribute(&child, "default").as_deref() == Some("true")
                    && let Some(uri) = attribute(&child, "uri")
                {
                    environment.default_collation = Some(uri);
                }
            }
            "static-base-uri" => match attribute(&child, "uri") {
                // The catalogue's sentinel for "no base URI is available", not a literal one.
                Some(uri) if uri == "#UNDEFINED" => environment.explicit_no_base_uri = true,
                uri => environment.base_uri = uri,
            },
            "schema" => environment.needs_schema = true,
            _ => {}
        }
    }
    environment
}

/// The expanded name a `<param name="…">` denotes.
///
/// An unprefixed name is in no namespace; a prefixed one is resolved against the environment's
/// namespace bindings, which [`read_environment`] has already folded a `<param>`'s own `xmlns:`
/// declarations into. Binding the *lexical* spelling instead would leave `$test:x` — which the
/// query resolves to its expanded name before looking it up — unsupplied.
fn parameter_name(lexical: &str, namespaces: &[(String, String)]) -> ExpandedName {
    match lexical.split_once(':') {
        None => ExpandedName::new(None::<String>, lexical),
        Some((prefix, local)) => {
            let uri = namespaces
                .iter()
                .find(|(bound, _)| bound == prefix)
                .map(|(_, uri)| uri.clone());
            ExpandedName::new(uri, local)
        }
    }
}

/// Resolves a lexical `QName` — like the `name` on a `<decimal-format>` — against `node`'s own
/// in-scope namespaces.
///
/// Unlike `fn:resolve-QName`, an unprefixed name is always in *no* namespace: the catalogue
/// itself declares a default `xmlns` (`http://www.w3.org/2010/09/qt-fots-catalog`) that every
/// element in the document inherits, and letting that leak into an unprefixed decimal-format name
/// would silently namespace-qualify names the catalogue writes as plain (`name="ch"`,
/// `name="decimal1"`, …) — exactly the names `fn:format-number`'s third argument is written to
/// look up unqualified. A `name` attribute is treated the way XSLT/XQuery treat every
/// attribute-valued `QName` outside of an actual element/type name: no default namespace applies.
fn resolve_local_qname(lexical: &str, node: &NodeRef) -> ExpandedName {
    match lexical.split_once(':') {
        None => ExpandedName::new(None::<String>, lexical),
        Some((prefix, local)) => {
            let uri = node.namespaces().into_iter().find_map(|namespace| {
                let bound = namespace.node_name()?;
                (bound.local_name() == prefix).then(|| namespace.string_value())
            });
            ExpandedName::new(uri, local)
        }
    }
}

/// The single-character value of attribute `local` on `node`, if present — every
/// [`DecimalFormat`] property but `infinity`/`NaN` is exactly one Unicode scalar value.
fn char_attribute(node: &NodeRef, local: &str) -> Option<char> {
    attribute(node, local)?.chars().next()
}

/// Reads a `<decimal-format>` element's properties (F&O 3.1 §4.7.1) onto a fresh
/// [`DecimalFormat`], leaving every property this element does not mention at the specification
/// default.
fn read_decimal_format(node: &NodeRef) -> DecimalFormat {
    let mut format = DecimalFormat::default();
    if let Some(value) = char_attribute(node, "decimal-separator") {
        format = format.with_decimal_separator(value);
    }
    if let Some(value) = char_attribute(node, "grouping-separator") {
        format = format.with_grouping_separator(value);
    }
    if let Some(value) = char_attribute(node, "exponent-separator") {
        format = format.with_exponent_separator(value);
    }
    if let Some(value) = attribute(node, "infinity") {
        format = format.with_infinity(value);
    }
    if let Some(value) = char_attribute(node, "minus-sign") {
        format = format.with_minus_sign(value);
    }
    if let Some(value) = attribute(node, "NaN") {
        format = format.with_nan(value);
    }
    if let Some(value) = char_attribute(node, "percent") {
        format = format.with_percent(value);
    }
    if let Some(value) = char_attribute(node, "per-mille") {
        format = format.with_per_mille(value);
    }
    if let Some(value) = char_attribute(node, "zero-digit") {
        format = format.with_zero_digit(value);
    }
    if let Some(value) = char_attribute(node, "digit") {
        format = format.with_digit(value);
    }
    if let Some(value) = char_attribute(node, "pattern-separator") {
        format = format.with_pattern_separator(value);
    }
    format
}

/// Rewrites every file path a catalogue-*root*-level [`Environment`] captured to an absolute path
/// under `root`.
///
/// `catalog.xml` declares thirteen shared `<environment>`s at its own root, referenced by `ref`
/// from test-sets scattered all across the tree (`docs/works.xml`, relative to the *catalogue*
/// root, not to whichever test-set directory happens to reference it). Left relative, the same
/// `docs/works.xml` resolved against e.g. `prod/`'s own directory would look for
/// `prod/docs/works.xml`, which does not exist. Absolutizing immediately, once, when the shared
/// environment is first read, means [`resolve_environment`] can keep resolving every other
/// (test-set-local) source the same way it always has: `PathBuf::join` leaves an already-absolute
/// path alone, so no branch is needed at the point of use.
fn absolutize(environment: &mut Environment, root: &Path) {
    for source in &mut environment.sources {
        source.file = root.join(&source.file).to_string_lossy().into_owned();
    }
    for (_, file) in &mut environment.resources {
        *file = root.join(&file).to_string_lossy().into_owned();
    }
    for (_, files) in &mut environment.collections {
        for file in files {
            *file = root.join(&file).to_string_lossy().into_owned();
        }
    }
}

/// The catalogue's own root-level `<environment>` definitions — `catalog.xml`'s direct children
/// — shared by every test-set that references one by name without a local definition of its own.
/// See [`absolutize`] for why every file path is resolved immediately.
#[must_use]
pub(crate) fn shared_environments(
    catalog_root: &NodeRef,
    root: &Path,
) -> BTreeMap<String, Environment> {
    let mut environments = BTreeMap::new();
    for definition in children_named(catalog_root, "environment") {
        if let Some(name) = attribute(&definition, "name") {
            let mut environment = read_environment(&definition);
            absolutize(&mut environment, root);
            environments.insert(name, environment);
        }
    }
    environments
}

/// The documents a test-set's environment makes available to `fn:doc`/`fn:collection`, plus the
/// text resources `fn:unparsed-text`/`fn:json-doc` read — see the module documentation.
#[derive(Debug)]
pub(crate) struct Documents {
    by_uri: BTreeMap<String, NodeRef>,
    text_by_uri: BTreeMap<String, Vec<u8>>,
    /// Named collections (the empty string is the default collection): each member URI is also a
    /// key in `by_uri`.
    collections: BTreeMap<String, Vec<String>>,
    /// Collections whose members are arbitrary items, from an `<collection><query>` declaration
    /// (see [`Environment::collection_queries`]). Consulted before `collections`, which can only
    /// hold documents.
    item_collections: BTreeMap<String, Vec<Item>>,
    /// The test-set directory, consulted when a URI names a file nothing pre-registered — e.g. a
    /// JSON resource path written directly into a query body.
    directory: PathBuf,
    /// Whether an unregistered URI may fall back to a read from `directory` at all.
    ///
    /// `true` by default — most environments declare no `<static-base-uri>`, and the JSON Test
    /// Suite cluster (and others) rely on exactly this fallback to reach a resource named by a
    /// bare relative path written directly into the query body. It is `false` only when the
    /// environment's `<static-base-uri>` spells the catalogue's `#UNDEFINED` sentinel: a
    /// deliberate assertion that *no* base URI is available, so a relative reference must fail to
    /// resolve rather than be quietly rescued from disk (`fn-unparsed-text-023` and its siblings
    /// exist specifically to test that failure).
    allow_fallback: bool,
}

impl Default for Documents {
    fn default() -> Self {
        Self {
            by_uri: BTreeMap::new(),
            text_by_uri: BTreeMap::new(),
            collections: BTreeMap::new(),
            item_collections: BTreeMap::new(),
            directory: PathBuf::new(),
            allow_fallback: true,
        }
    }
}

impl Documents {
    /// Reads a URI nothing pre-registered, trying the URI as written and then with its
    /// percent-escapes decoded.
    ///
    /// A URI's path is percent-*encoded*, so the file it names is the decoded one:
    /// `misc-JsonTestSuite/n_structure_trailing_%23` retrieves
    /// `…/n_structure_trailing_%23.json`, and the fixture on disk is named with a literal `#`
    /// — the character `%23` stands for. The literal spelling is tried first so that a file
    /// whose own name really does contain a per-cent sign still resolves.
    fn read_fallback(&self, uri: &str) -> Option<Vec<u8>> {
        if !self.allow_fallback {
            return None;
        }
        self.candidate_paths(uri)
            .into_iter()
            .find_map(|path| fs::read(path).ok())
    }

    /// The filesystem paths `uri` may name, in the order they should be tried.
    ///
    /// An absolute `file:` URI names its path outright and is *not* resolved against the test-set
    /// directory — that is the shape every relative reference takes once the engine has resolved
    /// it against the default static base URI (see [`apply_default_base_uri`]), so without this
    /// the whole fallback would stop working the moment a base URI exists. Anything else is a
    /// reference relative to the test-set's own directory, as it has always been.
    fn candidate_paths(&self, uri: &str) -> Vec<PathBuf> {
        let (base, reference): (&Path, &str) = match file_uri_path(uri) {
            Some(path) => (Path::new(""), path),
            None => (self.directory.as_path(), uri),
        };
        let mut paths = vec![base.join(reference)];
        if let Some(decoded) = percent_decode(reference) {
            paths.push(base.join(decoded));
        }
        paths
    }
}

impl DocumentResolver for Documents {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        if let Some(node) = self.by_uri.get(uri) {
            return Some(node.clone());
        }
        let bytes = self.read_fallback(uri)?;
        let text = decode_xml_bytes(&bytes)?;
        let options = ParseOptions::default().document_uri(uri.to_owned());
        let tree = Tree::parse_str_with(&text, options).ok()?;
        Some(tree.root())
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        let key = uri.unwrap_or("");
        let members = self.collections.get(key)?;
        Some(
            members
                .iter()
                .filter_map(|member| self.by_uri.get(member).cloned())
                .collect(),
        )
    }

    /// A collection declared by `<query>` answers its items directly; everything else is the
    /// document-only reading [`Documents::collection`] gives, lifted into items.
    fn collection_items(&self, uri: Option<&str>) -> Option<Vec<Item>> {
        if let Some(items) = self.item_collections.get(uri.unwrap_or("")) {
            return Some(items.clone());
        }
        Some(self.collection(uri)?.into_iter().map(Item::Node).collect())
    }

    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        self.unparsed_text_result(uri, encoding).ok()
    }

    fn unparsed_text_result(
        &self,
        uri: &str,
        encoding: Option<&str>,
    ) -> Result<String, TextRetrievalError> {
        let bytes = match self.text_by_uri.get(uri) {
            Some(bytes) => bytes.clone(),
            None => self
                .read_fallback(uri)
                .ok_or(TextRetrievalError::Unretrievable)?,
        };
        decode_resource(&bytes, encoding)
    }
}

/// Decodes the `%XX` escapes in a URI, or `None` when there are none to decode.
///
/// Only a per-cent sign followed by two hexadecimal digits is an escape; anything else is left
/// exactly as written, so a name that merely *contains* a per-cent sign survives unchanged. The
/// decoded octets are interpreted as UTF-8, which is what a URI's path is defined to encode.
fn percent_decode(uri: &str) -> Option<String> {
    if !uri.contains('%') {
        return None;
    }
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut decoded_any = false;
    while index < bytes.len() {
        let escape = (bytes[index] == b'%')
            .then(|| {
                let hex = uri.get(index + 1..index + 3)?;
                // `u8::from_str_radix` also accepts a leading sign; a URI escape is two
                // hexadecimal digits and nothing else.
                hex.bytes()
                    .all(|digit| digit.is_ascii_hexdigit())
                    .then(|| u8::from_str_radix(hex, 16).ok())
                    .flatten()
            })
            .flatten();
        match escape {
            Some(byte) => {
                out.push(byte);
                index += 3;
                decoded_any = true;
            }
            None => {
                out.push(bytes[index]);
                index += 1;
            }
        }
    }
    decoded_any.then(|| String::from_utf8(out).ok()).flatten()
}

/// Decodes a retrieved resource's bytes to text, honouring — in priority order — a byte-order
/// mark, an explicit `$encoding` argument, and finally the XML/text default of UTF-8.
///
/// A byte-order mark is authoritative: F&O 3.1 §14.8 lets it override a requested `$encoding`,
/// and the QT3 catalogue exercises exactly that (`fn-unparsed-text-043`/`-044` retrieve a
/// `UTF-16LE`/`UTF-16BE`-with-BOM resource via the *one-argument* `fn:unparsed-text($href)`,
/// which supplies no `$encoding` at all — the BOM is the only signal there is).
///
/// When neither a byte-order mark nor an explicit encoding pins anything down and the UTF-8
/// default fails to decode, this is reported as [`TextRetrievalError::UnsupportedEncoding`]
/// (`FOUT1190`) rather than [`TextRetrievalError::InvalidOctets`] (`FOUT1200`): nothing
/// identified an encoding to fail *against*, so "no encoding could be determined" is the honest
/// reading. This exactly matches how the catalogue itself grades the case: `fn-unparsed-text-045`
/// retrieves an ISO-8859-1 resource with no declared encoding and accepts either the correctly
/// -guessed text or `FOUT1190` — never `FOUT1200`.
fn decode_resource(bytes: &[u8], encoding: Option<&str>) -> Result<String, TextRetrievalError> {
    let (target, skip, defaulted) = if let Some((encoding, len)) = detect_bom(bytes) {
        (encoding, len, false)
    } else if let Some(label) = encoding {
        let Some(encoding) = Encoding::for_label(label) else {
            return Err(TextRetrievalError::UnsupportedEncoding);
        };
        (encoding, 0, false)
    } else {
        (Encoding::Utf8, 0, true)
    };
    let Some(payload) = bytes.get(skip..) else {
        return Err(TextRetrievalError::Unretrievable);
    };
    let mut decoder = target.new_decoder();
    let mut text = String::with_capacity(payload.len());
    match decoder.decode_to_string(payload, &mut text, true) {
        Ok(_) => Ok(text),
        Err(_) if defaulted => Err(TextRetrievalError::UnsupportedEncoding),
        Err(_) => Err(TextRetrievalError::InvalidOctets),
    }
}

/// Everything [`resolve_environment`] assembles from an [`Environment`] that neither driver's own
/// static-context type changes: the document/collection resolver, the context item, the
/// `<source role="$name">` variable bindings, and the trees that must outlive the evaluation.
pub(crate) struct ResolvedEnvironment {
    pub(crate) documents: std::sync::Arc<Documents>,
    pub(crate) context_item: Option<NodeRef>,
    /// `(name, value)` bindings from every `<source role="$name">` — bound both as a declared
    /// variable name (by the caller, on the static context) and, here, as its dynamic value.
    pub(crate) source_variables: Vec<(String, NodeRef)>,
    pub(crate) trees: Vec<Tree>,
}

/// Resolves `environment`'s sources, resources and collections into a [`ResolvedEnvironment`],
/// relative to `directory` (a test-set's own directory, or ignored for a path [`absolutize`]
/// already made absolute). `None` when the environment needs schema awareness (neither engine
/// has any) or a declared source/collection member could not be read and parsed.
pub(crate) fn resolve_environment(
    environment: &Environment,
    directory: &Path,
) -> Option<ResolvedEnvironment> {
    if environment.needs_schema {
        return None;
    }

    let mut documents = Documents {
        directory: directory.to_path_buf(),
        allow_fallback: !environment.explicit_no_base_uri,
        ..Documents::default()
    };
    let mut trees = Vec::new();
    let mut context_item = None;
    let mut source_variables = Vec::new();

    // Every declared resource and source is registered under three keys: the reference as the
    // catalogue wrote it, the absolute path, and the absolute `file:` URI. The third is what a
    // *relative* reference in a query body becomes once the engine has resolved it against the
    // default static base URI (see `apply_default_base_uri`) — without it, a document the
    // environment declares would be re-read from disk instead of found, and `fn:doc('x') is .`
    // would compare two separately parsed copies of the same file.
    for (uri, file) in &environment.resources {
        let path = directory.join(file);
        if let Ok(bytes) = fs::read(&path) {
            documents.text_by_uri.insert(uri.clone(), bytes.clone());
            documents
                .text_by_uri
                .insert(path.to_string_lossy().into_owned(), bytes.clone());
            documents.text_by_uri.insert(file_uri(&path), bytes.clone());
            if let Some(key) = resolved_relative_key(directory, uri) {
                documents.text_by_uri.insert(key, bytes);
            }
        }
    }

    for source in &environment.sources {
        let path = directory.join(&source.file);
        let tree = parse_source(&path)?;
        let root = tree.root();
        if source.role == "." {
            context_item = Some(root.clone());
        } else if let Some(name) = source.role.strip_prefix('$') {
            source_variables.push((name.to_owned(), root.clone()));
        }
        documents.by_uri.insert(source.file.clone(), root.clone());
        documents
            .by_uri
            .insert(path.to_string_lossy().into_owned(), root.clone());
        documents.by_uri.insert(file_uri(&path), root.clone());
        if let Some(uri) = &source.uri {
            documents.by_uri.insert(uri.clone(), root.clone());
            if let Some(key) = resolved_relative_key(directory, uri) {
                documents.by_uri.insert(key, root.clone());
            }
        }
        trees.push(tree);
    }

    for (collection_uri, files) in &environment.collections {
        let mut member_uris = Vec::with_capacity(files.len());
        for file in files {
            let path = directory.join(file);
            // A member is registered under its `file:` URI *and* under the plain path, so a
            // `document-uri`/`doc` round trip resolves (see `file_uri`) without taking away the
            // path key an environment that names the same file elsewhere still looks it up by.
            let uri = file_uri(&path);
            let tree = parse_source_as(&path, &uri)?;
            documents.by_uri.insert(uri.clone(), tree.root());
            documents
                .by_uri
                .insert(path.to_string_lossy().into_owned(), tree.root());
            member_uris.push(uri);
            trees.push(tree);
        }
        documents
            .collections
            .insert(collection_uri.clone(), member_uris);
    }

    // A `<collection><query>` is evaluated against the resolver as built so far, because that is
    // where its inputs are: `app/UseCaseR31.xml` builds its collections with
    // `unparsed-text-lines("UseCaseR31/users.json") ! parse-json(.)` over a resource this very
    // environment declares. The resolver is shared as an `Arc` because that is what an evaluation
    // context takes; every clone is dropped by the time the collections have been evaluated, which
    // is what lets the resolver be completed with them afterwards.
    let mut documents = std::sync::Arc::new(documents);
    let evaluated: Vec<(String, Vec<Item>)> = environment
        .collection_queries
        .iter()
        .filter_map(|(uri, query)| {
            Some((
                uri.clone(),
                collection_query_items(query, environment, &documents)?,
            ))
        })
        .collect();
    if let Some(resolver) = std::sync::Arc::get_mut(&mut documents) {
        resolver.item_collections.extend(evaluated);
    }

    Some(ResolvedEnvironment {
        documents,
        context_item,
        source_variables,
        trees,
    })
}

/// Evaluates one `<collection><query>` into the item sequence the collection holds.
///
/// The expression is plain XPath 3.1 — the catalogue's collection queries are `1 to 10`,
/// `("goodbye", "cruel", "world")` and `unparsed-text-lines(…) ! parse-json(.)`, none of which
/// needs anything XQuery adds — and it is evaluated in the environment's own namespaces and static
/// base URI, so a relative resource reference in it resolves exactly as it would inside a case.
///
/// `None` when the expression does not compile or raises, and equally when it yields a bare
/// function item: [`Item`] models a node, an atomic value, a map or an array, so a sequence
/// containing a general function could not be stored faithfully, and reporting the collection as
/// absent (`FODC0002`) is honest where dropping the item silently would not be.
fn collection_query_items(
    query: &str,
    environment: &Environment,
    documents: &std::sync::Arc<Documents>,
) -> Option<Vec<Item>> {
    let mut statik = StaticContext::new();
    for (prefix, uri) in &environment.namespaces {
        statik = statik.with_namespace(prefix.clone(), uri.clone());
    }
    if let Some(base) = &environment.base_uri {
        statik = statik.with_base_uri(base.clone());
    }
    let compiled = XPath::compile(query, XPathVersion::XPath31, statik).ok()?;
    let resolver: std::sync::Arc<dyn DocumentResolver> = documents.clone();
    let context = compiled.context().with_document_resolver(resolver);
    compiled
        .evaluate_value_in(&context)
        .ok()?
        .into_items()
        .into_iter()
        .map(oxixml_xpath::XpathItem::into_item)
        .collect()
}

/// The specification version both engines implement, as the catalogue spells it: XPath 3.1 and
/// XQuery 3.1, written `31` in every `spec` token (`XP31`, `XQ31+`, …).
const IMPLEMENTED_SPEC_VERSION: u32 = 31;

/// Whether a `<dependency type="spec" value="…">` is satisfied by an engine implementing exactly
/// [`IMPLEMENTED_SPEC_VERSION`] of `family` (`"XP"` for XPath, `"XQ"` for XQuery).
///
/// A `spec` dependency lists the specification versions a case *applies to*, and a driver runs the
/// case only when its own version is among them. The list is a set of alternatives, each either an
/// exact version (`XP20`) or an open-ended one (`XP20+`, "this version or any later"), so:
///
/// * `XP31`, `XP31+`, `XP30+`, `XP20+` are all satisfied by an XPath 3.1 engine;
/// * `XP20`, `XP30` alone are **not** — they name behaviour that a later version deliberately
///   changed, which is why the catalogue so often writes a case twice, once pinned to the older
///   versions and once to `XP31+`. `fn/string-join.xml` is the clearest example:
///   `fn-string-join-27` (`XP20 XQ10 XP30 XQ30`) wants `fn:string-join(1 to 5, "")` to raise
///   `XPTY0004`, because the function took `xs:string*` before 3.1; `fn-string-join-28`
///   (`XP31+ XQ31+`) wants `"12345"` from the very same expression. Running the first against a
///   3.1 function library measures nothing about conformance.
/// * a token of another family (an `XQ…` token seen by the XPath driver, or the `XT30+` an
///   `fn:transform`-flavoured case adds) contributes nothing either way.
fn spec_satisfied(value: &str, family: &str) -> bool {
    value
        .split_whitespace()
        .any(|token| spec_token_covers(token, family))
}

/// Whether one `spec` token names a version of `family` that an engine implementing exactly
/// [`IMPLEMENTED_SPEC_VERSION`] provides. See [`spec_satisfied`].
fn spec_token_covers(token: &str, family: &str) -> bool {
    let (version, open_ended) = match token.strip_suffix('+') {
        Some(rest) => (rest, true),
        None => (token, false),
    };
    let Some(digits) = version.strip_prefix(family) else {
        return false;
    };
    let Ok(version) = digits.parse::<u32>() else {
        return false;
    };
    if open_ended {
        version <= IMPLEMENTED_SPEC_VERSION
    } else {
        version == IMPLEMENTED_SPEC_VERSION
    }
}

/// The version of XML both engines operate at, as an `xml-version` dependency spells one.
///
/// XML 1.0, and the catalogue's pairs of cases are unanimous about it: `fn:codepoints-to-string`
/// raises `FOCH0001` for a C0 control character, which is XML 1.0's `Char` production and the
/// answer `K2-Serialization-7a` requires (its twin `-7` requires the string to be built and
/// serialized as character references, which only XML 1.1 admits); an XQuery direct constructor
/// that *undeclares* a prefix is `XQST0085`, which the specification defines as the answer of a
/// processor that does not support Namespaces in XML 1.1; and `#x85`/`#x2028` in query text are
/// ordinary characters rather than line endings.
///
/// This says nothing about which documents the parser reads or which declaration the serializer
/// writes — `oxixml-infoset` does parse a document declaring `version="1.1"`, and the serializer
/// does write `version="1.1"` output with prefix undeclarations. The dependency asks which
/// version the *processor* operates at, and these are the properties the catalogue tests it by.
const IMPLEMENTED_XML_VERSION: &str = "1.0";

/// The edition of [`IMPLEMENTED_XML_VERSION`] the name productions follow: the fifth, whose
/// `NameStartChar` admits code points (`#x37F` among them) that the fourth edition excluded.
const IMPLEMENTED_XML_EDITION: u32 = 5;

/// Whether a `<dependency type="xml-version" value="…">` names the version [this processor
/// operates at](IMPLEMENTED_XML_VERSION).
///
/// The value is a whitespace-separated set of alternatives, and a processor answers for exactly
/// one version, because the catalogue writes the two answers as *pairs* over the same query —
/// `K2-Serialization-7` requires `codepoints-to-string(1 to 31)` to serialize as character
/// references under 1.1 while `-7a` requires `FOCH0001` under 1.0 — so claiming both would run
/// both halves of every pair and count one a failure whatever the processor does.
fn xml_version_satisfied(value: &str) -> bool {
    value.split_whitespace().any(xml_version_token_covers)
}

/// Whether one `xml-version` token names the version and edition this processor operates at.
///
/// A token is a version, optionally followed by `:` and an edition constraint: `1.0`, `1.1`,
/// `1.0:5+` ("fifth edition or later") and `1.0:4-` ("fourth edition or earlier") are the four
/// the catalogue's own schema defines. A token with no edition constraint names every edition of
/// that version.
fn xml_version_token_covers(token: &str) -> bool {
    let Some((version, edition)) = token.split_once(':') else {
        return token == IMPLEMENTED_XML_VERSION;
    };
    if version != IMPLEMENTED_XML_VERSION {
        return false;
    }
    if let Some(digits) = edition.strip_suffix('+') {
        return digits
            .parse::<u32>()
            .is_ok_and(|edition| IMPLEMENTED_XML_EDITION >= edition);
    }
    if let Some(digits) = edition.strip_suffix('-') {
        return digits
            .parse::<u32>()
            .is_ok_and(|edition| IMPLEMENTED_XML_EDITION <= edition);
    }
    edition
        .parse::<u32>()
        .is_ok_and(|edition| IMPLEMENTED_XML_EDITION == edition)
}

/// Whether every dependency declared on `node` is one this driver can decide, given the family
/// of specification versions it implements (`"XP"` for XPath, `"XQ"` for XQuery — the catalogue
/// spells its `spec` tokens `XP20`, `XP30`, `XQ10`, `XQ31`, …) and the feature names it
/// implements.
fn dependencies_supported(node: &NodeRef, spec_family: &str, supported_features: &[&str]) -> bool {
    for dependency in children_named(node, "dependency") {
        let kind = attribute(&dependency, "type").unwrap_or_default();
        let value = attribute(&dependency, "value").unwrap_or_default();
        let satisfied =
            attribute(&dependency, "satisfied").is_none_or(|text| text.trim() != "false");
        let holds = match kind.as_str() {
            "spec" => spec_satisfied(&value, spec_family),
            "feature" => supported_features.contains(&value.as_str()),
            "xml-version" => xml_version_satisfied(&value),
            // an undecidable dependency kind (e.g. `xsd-version`, `unicode-version`)
            _ => return false,
        };
        if holds != satisfied {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_environment_collects_every_child_kind() {
        let tree = Tree::parse_str(
            r#"<environment name="e">
                <source role="." file="in.xml" uri="urn:in"/>
                <namespace prefix="a" uri="urn:a"/>
                <param name="p" select="1"/>
                <resource file="r.txt" uri="urn:r"/>
                <static-base-uri uri="urn:base"/>
            </environment>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let environment = read_environment(&root);
        assert_eq!(environment.sources.len(), 1);
        assert_eq!(environment.sources[0].role, ".");
        assert_eq!(environment.sources[0].file, "in.xml");
        assert_eq!(environment.sources[0].uri.as_deref(), Some("urn:in"));
        assert_eq!(
            environment.namespaces,
            vec![("a".to_owned(), "urn:a".to_owned())]
        );
        assert_eq!(
            environment.parameters,
            vec![("p".to_owned(), "1".to_owned())]
        );
        assert_eq!(
            environment.resources,
            vec![("urn:r".to_owned(), "r.txt".to_owned())]
        );
        assert_eq!(environment.base_uri.as_deref(), Some("urn:base"));
        assert!(!environment.needs_schema);
    }

    /// `prod/VarDecl.external.xml`'s `extvardeclwithouttype-24`/`extvardeclwithtype-24` declare
    /// their parameter's prefix on the `<param>` element itself, so the binding has to be taken
    /// from there — otherwise the case's `$test:x` names a variable nothing supplied.
    #[test]
    fn a_param_prefix_declared_on_the_param_element_becomes_a_namespace_binding() {
        let tree = Tree::parse_str(
            r#"<environment>
                <param name="test:x" xmlns:test="urn:example:test" select="2" declared="true"/>
               </environment>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let environment = read_environment(&root);
        assert_eq!(
            environment.namespaces,
            vec![("test".to_owned(), "urn:example:test".to_owned())]
        );
        assert_eq!(
            environment.parameters,
            vec![("test:x".to_owned(), "2".to_owned())]
        );
    }

    /// An explicit `<namespace>` for the same prefix is the environment's own declaration and is
    /// not duplicated or overridden by the one on a `<param>`.
    #[test]
    fn an_explicit_namespace_declaration_is_not_shadowed_by_a_param_prefix() {
        let tree = Tree::parse_str(
            r#"<environment>
                <namespace prefix="test" uri="urn:example:declared"/>
                <param name="test:x" xmlns:test="urn:example:other" select="2"/>
               </environment>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let environment = read_environment(&root);
        assert_eq!(
            environment.namespaces,
            vec![("test".to_owned(), "urn:example:declared".to_owned())]
        );
    }

    #[test]
    fn a_context_item_child_is_read_as_a_selector() {
        let tree =
            Tree::parse_str(r#"<environment><context-item select="'London'"/></environment>"#)
                .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            read_environment(&root).context_item_select.as_deref(),
            Some("'London'")
        );
    }

    /// `fn/collection.xml` declares three collections that hold no documents at all.
    #[test]
    fn a_collection_declared_by_query_is_read_and_evaluated_into_items() {
        let tree = Tree::parse_str(
            r#"<environment>
                <collection uri="urn:example:integers"><query>1 to 10</query></collection>
                <collection uri=""><query>("goodbye", "cruel", "world")</query></collection>
               </environment>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let environment = read_environment(&root);
        assert_eq!(environment.collection_queries.len(), 2);
        assert!(
            environment.collections.is_empty(),
            "a query collection has no file members"
        );

        let resolved =
            resolve_environment(&environment, Path::new(".")).expect("nothing to read from disk");
        let integers = resolved
            .documents
            .collection_items(Some("urn:example:integers"))
            .expect("the named collection was declared");
        assert_eq!(integers.len(), 10);
        assert_eq!(
            integers[0].as_atomic().map(ToString::to_string),
            Some("1".to_owned()),
            "a collection may hold atomic values, not only nodes"
        );
        let default = resolved
            .documents
            .collection_items(None)
            .expect("the default collection was declared");
        assert_eq!(default.len(), 3);
        assert!(
            resolved
                .documents
                .collection_items(Some("urn:example:absent"))
                .is_none()
        );
    }

    #[test]
    fn a_schema_child_marks_the_environment_as_needing_schema_awareness() {
        let tree = Tree::parse_str(r#"<environment><schema/></environment>"#)
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert!(read_environment(&root).needs_schema);
    }

    #[test]
    fn dependencies_supported_matches_a_spec_the_engine_implements() {
        let tree = Tree::parse_str(
            r#"<test-case><dependency type="spec" value="XP31 XQ31"/></test-case>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert!(dependencies_supported(&root, "XP", &[]));
        assert!(dependencies_supported(&root, "XQ", &[]));
        assert!(!dependencies_supported(&root, "ZZ", &[]));
    }

    /// A case pinned to versions this engine does not implement is *not* runnable: see
    /// [`spec_satisfied`] for why running it anyway measures nothing. `fn-string-join-27` and
    /// `K-ErrorFunc-4` are the two the catalogue makes clearest.
    #[test]
    fn dependencies_supported_rejects_a_spec_naming_only_older_versions() {
        for value in ["XP20 XQ10 XP30 XQ30", "XQ10 XP20 XQ30 XP30"] {
            let tree = Tree::parse_str(&format!(
                r#"<test-case><dependency type="spec" value="{value}"/></test-case>"#
            ))
            .expect("well-formed fixture");
            let root = tree.document_element().expect("a document element");
            assert!(!dependencies_supported(&root, "XP", &[]), "{value}");
            assert!(!dependencies_supported(&root, "XQ", &[]), "{value}");
        }
    }

    /// Every distinct `<dependency type="spec">` value the catalogue actually writes, decided for
    /// both engine families. Enumerating the real corpus rather than invented tokens is what makes
    /// this a regression test: a rule that is right on `XP31` and wrong on `XQ10+` would skip
    /// four thousand cases and still look correct on a hand-picked example.
    #[test]
    fn every_spec_value_the_catalogue_writes_is_decided() {
        // (value, satisfied for "XP", satisfied for "XQ")
        let corpus: &[(&str, bool, bool)] = &[
            ("XQ10+", false, true),
            ("XQ30+", false, true),
            ("XP31+ XQ31+", true, true),
            ("XQ31+", false, true),
            ("XP30+ XQ30+", true, true),
            ("XP20+ XQ10+", true, true),
            ("XP30+ XQ10+", true, true),
            ("XQ31", false, true),
            ("XQ10", false, false),
            ("XP31 XQ31", true, true),
            ("XP20+", true, false),
            ("XP30+", true, false),
            ("XQ30+ XP30+", true, true),
            ("XQ10+ XP20+", true, true),
            ("XP20 XQ10", false, false),
            ("XP31+ XQ31+ XT30+", true, true),
            ("XQ10 XQ30", false, false),
            ("XQ10+ XP30+", true, true),
            ("XQ10 XP20", false, false),
            ("XP20 XP30 XQ10 XQ30", false, false),
            ("XP20 XP30 XP31 XQ10 XQ30 XQ31", true, true),
            ("XQ31+ XP31+", true, true),
            ("XP31+", true, false),
            ("XP20 XQ10 XQ30 XP30", false, false),
            ("XP31+ XQ31", true, true),
            ("XP30 XQ30", false, false),
            ("XQ30 XP30", false, false),
            ("XQ30", false, false),
            ("XP20 XQ10 XP30 XQ30", false, false),
            ("XP20", false, false),
        ];
        for (value, xpath, xquery) in corpus {
            assert_eq!(spec_satisfied(value, "XP"), *xpath, "XP: `{value}`");
            assert_eq!(spec_satisfied(value, "XQ"), *xquery, "XQ: `{value}`");
        }
    }

    #[test]
    fn a_spec_token_is_open_ended_only_with_a_plus() {
        assert!(spec_token_covers("XP31", "XP"));
        assert!(spec_token_covers("XP31+", "XP"));
        assert!(spec_token_covers("XP20+", "XP"), "3.1 is later than 2.0");
        assert!(
            !spec_token_covers("XP20", "XP"),
            "exactly 2.0, which we are not"
        );
        assert!(!spec_token_covers("XQ31", "XP"), "wrong family");
        assert!(
            !spec_token_covers("XT30+", "XP"),
            "not an XPath token at all"
        );
        assert!(!spec_token_covers("XP", "XP"), "no version at all");
        assert!(!spec_token_covers("XPxx", "XP"), "not a version number");
    }

    /// Every distinct `<dependency type="xml-version">` value the catalogue writes, decided the
    /// way [`IMPLEMENTED_XML_VERSION`] answers. The four are a closed set (the catalogue's own
    /// schema enumerates the token grammar), and they come in *pairs* over one query, so a rule
    /// that admitted two of them would run both halves of a pair and fail one.
    #[test]
    fn every_xml_version_value_the_catalogue_writes_is_decided() {
        let corpus: &[(&str, bool)] = &[
            ("1.0", true),
            ("1.1", false),
            // Fifth edition or later: this processor's names follow the fifth.
            ("1.0:5+ 1.1", true),
            // Fourth edition or earlier: `#x37F` may not start a name, which is not this one.
            ("1.0:4-", false),
        ];
        for (value, expected) in corpus {
            assert_eq!(xml_version_satisfied(value), *expected, "`{value}`");
        }
    }

    #[test]
    fn an_xml_version_token_reads_its_edition_constraint() {
        assert!(xml_version_token_covers("1.0"), "every edition of 1.0");
        assert!(xml_version_token_covers("1.0:5+"));
        assert!(xml_version_token_covers("1.0:4+"), "later than the fourth");
        assert!(xml_version_token_covers("1.0:5"), "exactly the fifth");
        assert!(!xml_version_token_covers("1.0:4"), "exactly the fourth");
        assert!(!xml_version_token_covers("1.0:6+"), "no sixth edition here");
        assert!(!xml_version_token_covers("1.0:4-"));
        assert!(xml_version_token_covers("1.0:5-"), "the fifth or earlier");
        assert!(!xml_version_token_covers("1.1"), "another version");
        assert!(!xml_version_token_covers("1.1:5+"), "another version");
        assert!(!xml_version_token_covers("1.0:"), "no edition at all");
        assert!(!xml_version_token_covers("1.0:x+"), "not an edition number");
    }

    /// An `xml-version` dependency is decided rather than making the case undecidable, and its
    /// `satisfied="false"` spelling inverts it like every other kind's.
    #[test]
    fn dependencies_supported_decides_an_xml_version() {
        for (value, satisfied, expected) in [
            ("1.0", "true", true),
            ("1.1", "true", false),
            ("1.1", "false", true),
            ("1.0", "false", false),
        ] {
            let tree = Tree::parse_str(&format!(
                r#"<test-case><dependency type="xml-version" value="{value}" satisfied="{satisfied}"/></test-case>"#
            ))
            .expect("well-formed fixture");
            let root = tree.document_element().expect("a document element");
            assert_eq!(
                dependencies_supported(&root, "XQ", &[]),
                expected,
                "{value} satisfied={satisfied}"
            );
        }
    }

    #[test]
    fn dependencies_supported_honours_satisfied_false() {
        let tree = Tree::parse_str(r#"<test-case><dependency type="feature" value="schemaImport" satisfied="false"/></test-case>"#).expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        // `schemaImport` is unsupported (empty feature list) but the case wants it *unsatisfied*.
        assert!(dependencies_supported(&root, "XP", &[]));
    }

    #[test]
    fn dependencies_supported_rejects_an_undecidable_kind() {
        let tree = Tree::parse_str(
            r#"<test-case><dependency type="unicode-version" value="6.0"/></test-case>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert!(!dependencies_supported(&root, "XP", &[]));
    }

    #[test]
    fn a_source_uri_registers_the_document_under_that_key_too() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-source-uri-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("in.xml"), "<a/>").expect("fixture file");
        let mut environment = Environment::default();
        environment.sources.push(Source {
            role: ".".to_owned(),
            file: "in.xml".to_owned(),
            uri: Some("urn:example:doc".to_owned()),
        });
        let resolved =
            resolve_environment(&environment, &directory).expect("no schema, a readable source");
        assert!(resolved.documents.document("urn:example:doc").is_some());
        assert!(resolved.documents.document("in.xml").is_some());
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn a_dollar_role_becomes_a_source_variable() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-role-var-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("in.xml"), "<a/>").expect("fixture file");
        let mut environment = Environment::default();
        environment.sources.push(Source {
            role: "$doc".to_owned(),
            file: "in.xml".to_owned(),
            uri: None,
        });
        let resolved =
            resolve_environment(&environment, &directory).expect("no schema, a readable source");
        assert!(resolved.context_item.is_none(), "no `.` role was declared");
        assert_eq!(resolved.source_variables.len(), 1);
        assert_eq!(resolved.source_variables[0].0, "doc");
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn collection_returns_none_when_nothing_is_registered() {
        let environment = Environment::default();
        let resolved = resolve_environment(&environment, Path::new("."))
            .expect("the empty environment always reproduces");
        assert!(resolved.documents.collection(None).is_none());
        assert!(resolved.documents.collection(Some("urn:x")).is_none());
    }

    #[test]
    fn a_named_collection_resolves_its_declared_members() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-collection-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("a.xml"), "<a/>").expect("fixture file");
        fs::write(directory.join("b.xml"), "<b/>").expect("fixture file");
        let mut environment = Environment::default();
        environment
            .collections
            .push((String::new(), vec!["a.xml".to_owned(), "b.xml".to_owned()]));
        let resolved =
            resolve_environment(&environment, &directory).expect("no schema, readable members");
        let members = resolved
            .documents
            .collection(None)
            .expect("the default collection was declared");
        assert_eq!(members.len(), 2);
        fs::remove_dir_all(&directory).ok();
    }

    /// `fn-collection/collection-009` is `for $doc in $c return doc(document-uri($doc))`, so a
    /// collection member's document URI has to be one `fn:doc` can resolve back to the very same
    /// node — an absolute `file:` URI, not a bare path that the environment's `http:` static base
    /// URI would swallow. The union in that case is over node *identity*, so a re-parse would
    /// double the count rather than fail outright.
    #[test]
    fn a_collection_member_round_trips_through_its_document_uri() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-collection-round-trip-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("a.xml"), "<a/>").expect("fixture file");
        let mut environment = Environment::default();
        environment
            .collections
            .push((String::new(), vec!["a.xml".to_owned()]));
        let resolved =
            resolve_environment(&environment, &directory).expect("no schema, a readable member");
        let members = resolved
            .documents
            .collection(None)
            .expect("the default collection was declared");
        let [member] = members.as_slice() else {
            panic!("one member was declared");
        };
        let uri = member.document_uri().expect("a member has a document URI");
        assert!(uri.starts_with("file://"), "{uri}");
        let again = resolved
            .documents
            .document(&uri)
            .expect("the member resolves under its own document URI");
        assert!(
            NodeRef::is_same_node(member, &again),
            "the round trip must give back the same node, not a re-parse"
        );
        // The path key every other environment shape looks a source up by still resolves.
        assert!(
            resolved
                .documents
                .document(&directory.join("a.xml").to_string_lossy())
                .is_some()
        );
        fs::remove_dir_all(&directory).ok();
    }

    /// `misc-JsonTestSuite/n_structure_trailing_%23` names its fixture with a percent-escape;
    /// the file on disk is named with the character the escape stands for.
    #[test]
    fn an_unregistered_resource_falls_back_to_the_percent_decoded_file_name() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-percent-escape-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("trailing_#.json"), "[1]").expect("fixture file");
        fs::write(directory.join("literal_%23.json"), "[2]").expect("fixture file");
        fs::write(directory.join("doc_#.xml"), "<a/>").expect("fixture file");
        let environment = Environment::default();
        let resolved = resolve_environment(&environment, &directory).expect("reproduces");
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("trailing_%23.json", None)
                .expect("the escape names the file on disk"),
            "[1]"
        );
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("literal_%23.json", None)
                .expect("a name that really contains `%23` still resolves literally"),
            "[2]"
        );
        assert!(resolved.documents.document("doc_%23.xml").is_some());
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("missing_%23.json", None)
                .expect_err("neither spelling is on disk"),
            TextRetrievalError::Unretrievable
        );
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn the_test_set_file_becomes_the_default_static_base_uri() {
        let mut environment = Environment::default();
        apply_default_base_uri(&mut environment, Path::new("/suite/prod/BaseURIDecl.xml"));
        assert_eq!(
            environment.base_uri.as_deref(),
            Some("file:///suite/prod/BaseURIDecl.xml")
        );

        // A declared base URI wins; the default only fills a gap.
        let mut declared = Environment {
            base_uri: Some("http://example.com/".to_owned()),
            ..Environment::default()
        };
        apply_default_base_uri(&mut declared, Path::new("/suite/prod/BaseURIDecl.xml"));
        assert_eq!(declared.base_uri.as_deref(), Some("http://example.com/"));

        // `#UNDEFINED` asserts that *no* base URI is available: the default must not undo it.
        let mut undefined = Environment {
            explicit_no_base_uri: true,
            ..Environment::default()
        };
        apply_default_base_uri(&mut undefined, Path::new("/suite/fn/unparsed-text.xml"));
        assert_eq!(undefined.base_uri, None);
    }

    /// With a `file:` static base URI in force every relative reference reaches the resolver
    /// already absolute, so the filesystem fallback has to understand that shape — including the
    /// percent-escaped fixture names `misc-JsonTestSuite` is built from.
    #[test]
    fn the_fallback_reads_an_absolute_file_uri() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-file-uri-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("data.json"), "[1]").expect("fixture file");
        fs::write(directory.join("trailing_#.json"), "[2]").expect("fixture file");
        fs::write(directory.join("doc.xml"), "<a/>").expect("fixture file");
        let resolved =
            resolve_environment(&Environment::default(), &directory).expect("reproduces");

        let absolute = file_uri(&directory.join("data.json"));
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result(&absolute, None)
                .expect("an absolute file: URI names its path outright"),
            "[1]"
        );
        let escaped = file_uri(&directory.join("trailing_%23.json"));
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result(&escaped, None)
                .expect("the escape names the file on disk"),
            "[2]"
        );
        assert!(
            resolved
                .documents
                .document(&file_uri(&directory.join("doc.xml")))
                .is_some()
        );
        assert!(
            resolved
                .documents
                .document(&file_uri(&directory.join("absent.xml")))
                .is_none()
        );
        fs::remove_dir_all(&directory).ok();
    }

    /// A relative `<resource uri="…">` is reachable both as declared and in the form the engine
    /// resolves it into — the shape `app/UseCaseJSON.xml` uses throughout (`uri="mildred.json"`
    /// for a file that actually lives under `UseCaseJSON/`).
    #[test]
    fn a_relative_declared_uri_is_registered_in_its_resolved_form_too() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-relative-uri-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(directory.join("nested")).expect("scratch directory");
        fs::write(directory.join("nested/data.json"), "[3]").expect("fixture file");
        fs::write(directory.join("nested/in.xml"), "<a/>").expect("fixture file");
        let mut environment = Environment::default();
        environment
            .resources
            .push(("data.json".to_owned(), "nested/data.json".to_owned()));
        environment.sources.push(Source {
            role: String::new(),
            file: "nested/in.xml".to_owned(),
            uri: Some("in.xml".to_owned()),
        });
        let resolved = resolve_environment(&environment, &directory).expect("reproduces");

        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("data.json", None)
                .expect("as declared"),
            "[3]"
        );
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result(&file_uri(&directory.join("data.json")), None)
                .expect("as the engine resolves it"),
            "[3]"
        );
        assert!(
            resolved
                .documents
                .document(&file_uri(&directory.join("in.xml")))
                .is_some()
        );
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn only_a_local_file_uri_names_a_path() {
        assert_eq!(file_uri_path("file:///a/b.xml"), Some("/a/b.xml"));
        assert_eq!(file_uri_path("file:///a/b.xml"), Some("/a/b.xml"));
        assert_eq!(file_uri_path("http://example.com/a"), None);
        assert_eq!(file_uri_path("file://relative"), None, "not rooted");
        assert_eq!(file_uri_path("b.xml"), None);
    }

    #[test]
    fn percent_decode_only_rewrites_real_escapes() {
        assert_eq!(percent_decode("a%23b").as_deref(), Some("a#b"));
        assert_eq!(percent_decode("%C3%A9").as_deref(), Some("é"));
        assert_eq!(percent_decode("plain"), None, "nothing to decode");
        assert_eq!(percent_decode("100%"), None, "a lone per-cent sign is text");
        assert_eq!(percent_decode("%zz"), None, "not hexadecimal");
        assert_eq!(
            percent_decode("%2Bx").as_deref(),
            Some("+x"),
            "a sign-looking escape is still an escape"
        );
        assert_eq!(percent_decode("%+1"), None, "but a sign is not a digit");
    }

    #[test]
    fn decode_resource_prefers_a_byte_order_mark_over_the_requested_encoding() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice("hi".as_bytes());
        // Requesting iso-8859-1 must lose to the UTF-8 BOM actually present.
        let text = decode_resource(&bytes, Some("iso-8859-1")).expect("valid UTF-8 past the BOM");
        assert_eq!(text, "hi");
    }

    #[test]
    fn decode_resource_defaults_to_utf8_and_reports_fout1190_not_fout1200() {
        // A lone high byte: invalid UTF-8, no BOM, no requested encoding.
        let bytes = [0xE9];
        let error = decode_resource(&bytes, None).expect_err("not valid UTF-8");
        assert_eq!(error, TextRetrievalError::UnsupportedEncoding);
    }

    #[test]
    fn decode_resource_reports_invalid_octets_against_an_explicit_encoding() {
        // 0xFF is not a valid US-ASCII byte.
        let bytes = [0xFF];
        let error = decode_resource(&bytes, Some("us-ascii")).expect_err("not valid US-ASCII");
        assert_eq!(error, TextRetrievalError::InvalidOctets);
    }

    #[test]
    fn decode_resource_rejects_an_unrecognized_explicit_encoding() {
        let error =
            decode_resource(b"hi", Some("utf-32")).expect_err("this crate has no UTF-32 decoder");
        assert_eq!(error, TextRetrievalError::UnsupportedEncoding);
    }

    /// Only `<collation default="true">` changes the static context; a bare `<collation>` merely
    /// declares a URI the processor should recognize, which the function library's own registry
    /// already does.
    #[test]
    fn only_a_default_collation_becomes_the_static_default() {
        let uca = "http://www.w3.org/2013/collation/UCA?lang=de;strength=primary";
        let tree = Tree::parse_str(&format!(
            r#"<environment><collation uri="{uca}" default="true"/></environment>"#
        ))
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            read_environment(&root).default_collation.as_deref(),
            Some(uca)
        );

        let tree = Tree::parse_str(&format!(
            r#"<environment><collation uri="{uca}"/></environment>"#
        ))
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(read_environment(&root).default_collation, None);
    }

    #[test]
    fn the_undefined_sentinel_is_not_treated_as_a_literal_base_uri() {
        let tree =
            Tree::parse_str(r##"<environment><static-base-uri uri="#UNDEFINED"/></environment>"##)
                .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let environment = read_environment(&root);
        assert_eq!(environment.base_uri, None);
        assert!(environment.explicit_no_base_uri);
    }

    #[test]
    fn an_explicit_undefined_base_uri_disables_the_filesystem_fallback() {
        // Regression: `fn-unparsed-text-023` and its siblings declare
        // `<static-base-uri uri="#UNDEFINED"/>` specifically to assert that a relative reference
        // is *unresolvable* — a resource that happens to exist on disk at that relative path
        // must not be quietly read anyway.
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-undefined-base-uri-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("present.xml"), "<a/>").expect("fixture file");
        fs::write(directory.join("present.txt"), "hello").expect("fixture file");
        let environment = Environment {
            explicit_no_base_uri: true,
            ..Environment::default()
        };
        let resolved = resolve_environment(&environment, &directory).expect("reproduces");
        assert!(
            resolved.documents.document("present.xml").is_none(),
            "the fallback must be disabled, even though the file genuinely exists"
        );
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("present.txt", None)
                .expect_err("the fallback must be disabled"),
            TextRetrievalError::Unretrievable
        );
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn document_falls_back_to_reading_the_test_set_directory() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-doc-fallback-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("unregistered.xml"), "<a/>").expect("fixture file");
        let environment = Environment::default();
        let resolved = resolve_environment(&environment, &directory).expect("reproduces");
        assert!(resolved.documents.document("unregistered.xml").is_some());
        assert!(resolved.documents.document("nope.xml").is_none());
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn unparsed_text_falls_back_to_reading_the_test_set_directory() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-text-fallback-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("scratch directory");
        fs::write(directory.join("data.json"), "[1,2,3]").expect("fixture file");
        let environment = Environment::default();
        let resolved = resolve_environment(&environment, &directory).expect("reproduces");
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("data.json", None)
                .expect("readable"),
            "[1,2,3]"
        );
        assert_eq!(
            resolved
                .documents
                .unparsed_text_result("missing.json", None)
                .expect_err("not on disk"),
            TextRetrievalError::Unretrievable
        );
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn resolve_local_qname_leaves_an_unprefixed_name_in_no_namespace_even_under_a_default_xmlns() {
        let tree = Tree::parse_str(r#"<decimal-format xmlns="urn:default" name="plain"/>"#)
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let resolved = resolve_local_qname("plain", &root);
        assert_eq!(resolved.namespace(), None);
        assert_eq!(resolved.local_name(), "plain");
    }

    #[test]
    fn resolve_local_qname_resolves_a_prefix_declared_on_the_element_itself() {
        let tree = Tree::parse_str(r#"<decimal-format xmlns:foo="urn:foo" name="foo:x"/>"#)
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let resolved = resolve_local_qname("foo:x", &root);
        assert_eq!(resolved.namespace(), Some("urn:foo"));
        assert_eq!(resolved.local_name(), "x");
    }

    #[test]
    fn read_decimal_format_reads_every_property() {
        let tree = Tree::parse_str(
            r##"<decimal-format decimal-separator="|" grouping-separator="_"
                                infinity="oo" minus-sign="~" NaN="nan"
                                percent="c" per-mille="m" zero-digit="0"
                                digit="#" pattern-separator=";" exponent-separator="E"/>"##,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let format = read_decimal_format(&root);
        assert_eq!(format.decimal_separator(), '|');
        assert_eq!(format.grouping_separator(), '_');
        assert_eq!(format.infinity(), "oo");
        assert_eq!(format.minus_sign(), '~');
        assert_eq!(format.nan(), "nan");
        assert_eq!(format.exponent_separator(), 'E');
    }

    #[test]
    fn absolutize_rewrites_source_and_resource_and_collection_paths() {
        let mut environment = Environment::default();
        environment.sources.push(Source {
            role: ".".to_owned(),
            file: "docs/a.xml".to_owned(),
            uri: None,
        });
        environment
            .resources
            .push(("urn:r".to_owned(), "docs/r.txt".to_owned()));
        environment
            .collections
            .push((String::new(), vec!["docs/c.xml".to_owned()]));
        absolutize(&mut environment, Path::new("/root"));
        assert_eq!(environment.sources[0].file, "/root/docs/a.xml");
        assert_eq!(environment.resources[0].1, "/root/docs/r.txt");
        assert_eq!(environment.collections[0].1[0], "/root/docs/c.xml");
    }

    #[test]
    fn shared_environments_reads_every_catalog_root_environment() {
        let tree = Tree::parse_str(
            r#"<catalog>
                <environment name="empty"/>
                <environment name="one"><source role="." file="a.xml"/></environment>
            </catalog>"#,
        )
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        let shared = shared_environments(&root, Path::new("/root"));
        assert_eq!(shared.len(), 2);
        assert_eq!(shared["one"].sources[0].file, "/root/a.xml");
    }
}
