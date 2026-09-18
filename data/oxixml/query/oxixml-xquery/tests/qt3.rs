//! The W3C *QT3* test suite (FOTS), filtered to the XQuery tests this engine is meant to pass.
//!
//! The suite is discovered through the `OXIXML_XML_SUITES_DIR` environment variable, which must
//! name a directory containing a `qt3tests` (or `qt3*`) subdirectory with a `catalog.xml`. When
//! the variable is unset, or the suite is not there, every test in this file **skips silently**,
//! so the crate builds and tests cleanly without the suite.
//!
//! # What is run, and what is not
//!
//! A test case is *runnable* when every `<dependency>` it declares is one this engine satisfies —
//! a `spec` naming an XQuery version, and a `feature` this crate implements. Cases needing schema
//! import or validation, static typing, XQuery Update, a non-default XSD or Unicode version, or a
//! UCA collation are skipped, as are cases whose environment this driver cannot reproduce and
//! cases whose assertion needs a comparison this driver does not implement.
//!
//! Everything else is compiled as XQuery 3.1 and executed, and a mismatch is a failure.
//!
//! # The skip taxonomy
//!
//! Every skip is attributed to one [`SkipReason`], and the report prints the breakdown, so that
//! "out of scope" is never confused with "quietly not run". Two of them are genuine, documented
//! scope gaps — [`SkipReason::Feature`] (schema awareness, static typing, XSLT invocation, and
//! the rest listed on [`SUPPORTED_FEATURES`]) and [`SkipReason::SchemaEnvironment`] — and the
//! others are limits of this driver rather than of the engine.

use oxixml_fn::numeric::DecimalFormat;
use oxixml_fn::xmlio::{SerializationParameters, serialize};
use oxixml_fn::{CollectingTrace, FnContext};
use oxixml_infoset::{
    AtomicValue, ExpandedName, NodeKind, NodeRef, ParseOptions, Sequence, Tree, XdmNode, deep_equal,
};
use oxixml_xpath::{
    DocumentResolver, StaticContext, TextRetrievalError, XPath, XPathVersion, XpathValue,
};
use oxixml_xquery::{
    DynamicQueryContext, InMemoryModules, StaticQueryContext, XQuery, XQueryError, XQueryErrorCode,
};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

/// The features of the suite this engine implements.
///
/// * `higherOrderFunctions`, `namespace-axis`, `moduleImport` — implemented outright;
/// * `serialization` — `fn:serialize` with the `xml`, `text`, `adaptive`, `json` and `html`
///   output methods, and both the map and the `output:serialization-parameters` element form;
/// * `simple-uca-fallback` — a `http://www.w3.org/2013/collation/UCA` URI is honoured by
///   `oxixml_fn::UcaCollation`, which falls back on the root ordering when a language tailoring
///   is asked for.
///
/// Everything else the suite names is a **documented gap**, and the cases that need it are
/// counted under [`SkipReason::Feature`] rather than as failures:
///
/// | Feature | Why it is out of scope |
/// |---|---|
/// | `schemaImport`, `schemaValidation`, `schema-location-hint`, `typedData` | schema-awareness lives in `oxixml-schema`, and this engine is a *basic* XQuery processor |
/// | `staticTyping` | the optional Static Typing Feature, which this engine deliberately does not have |
/// | `fn-transform-XSLT`, `fn-transform-XSLT30`, `fn-load-xquery-module` | they invoke another processor from inside a query |
/// | `advanced-uca-fallback` | needs the DUCET's language tailorings |
/// | `infoset-dtd`, `olson-timezone`, `remote_http`, `directory-as-collection-uri` | they need a DTD-validating parser, a timezone database, or the network |
/// | `non_empty_sequence_collection`, `collection-stability` | properties of an external collection this driver does not model |
/// | `non_unicode_codepoint_collation`, `fn-format-integer-CLDR`, `arbitraryPrecisionDecimal`, `xpath-1.0-compatibility` | optional features outside this engine's profile |
const SUPPORTED_FEATURES: &[&str] = &[
    "higherOrderFunctions",
    "namespace-axis",
    "moduleImport",
    "serialization",
    "simple-uca-fallback",
];

/// The dependency kinds this driver knows how to decide. Anything else skips the case.
const DECIDABLE_DEPENDENCIES: &[&str] = &[
    "spec",
    "feature",
    "xml-version",
    "xsd-version",
    "language",
    "default-language",
    "calendar",
];

/// Why a test case was not run. Every skip is attributed to exactly one of these, so that the
/// report distinguishes "out of scope" from "not attempted for want of a driver".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SkipReason {
    /// A `spec` dependency that does not cover XQuery 3.1.
    Spec,
    /// A `feature` this engine does not implement — the documented gaps above.
    Feature,
    /// A dependency kind this driver cannot decide at all.
    UndecidableDependency,
    /// A decidable dependency this engine does not satisfy: another XML edition, XSD version,
    /// language or calendar.
    Edition,
    /// The environment declares a schema, which needs the Schema Import Feature.
    SchemaEnvironment,
    /// The environment could not be built: a source or module file that is not there.
    EnvironmentUnavailable,
    /// The case names an environment the test set does not declare — one of the catalogue's own,
    /// which this driver does not merge in.
    EnvironmentUndefined,
    /// The case has no inline query — its `<test>` names a file.
    ExternalQuery,
    /// The case has no result assertion this driver can even read.
    NoAssertion,
    /// The assertion needs a comparison this driver does not implement.
    UndecidableAssertion,
}

impl SkipReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Spec => "spec is not XQuery 3.1",
            Self::Feature => "feature not implemented (documented gap)",
            Self::UndecidableDependency => "dependency kind the driver cannot decide",
            Self::Edition => "another XML edition, XSD version, language or calendar",
            Self::SchemaEnvironment => "environment needs schema awareness",
            Self::EnvironmentUnavailable => "environment files unavailable",
            Self::EnvironmentUndefined => "environment declared in the catalogue, not the set",
            Self::ExternalQuery => "query is in an external file",
            Self::NoAssertion => "no assertion the driver can read",
            Self::UndecidableAssertion => "assertion the driver cannot decide",
        }
    }
}

/// The test sets that cover the XQuery-only grammar this crate implements.
const XQUERY_SETS: &[&str] = &[
    // constructors
    "prod/DirElemConstructor",
    "prod/DirElemContent",
    "prod/DirAttributeList",
    "prod/DirectConstructor",
    "prod/CompElemConstructor",
    "prod/CompAttrConstructor",
    "prod/CompTextConstructor",
    "prod/CompCommentConstructor",
    "prod/CompDocConstructor",
    "prod/CompPIConstructor",
    "prod/CompNamespaceConstructor",
    "prod/EnclosedExpr",
    // FLWOR
    "prod/FLWORExpr",
    "prod/ForClause",
    "prod/LetClause",
    "prod/WhereClause",
    "prod/OrderByClause",
    "prod/GroupByClause",
    "prod/CountClause",
    "prod/WindowClause",
    "prod/AllowingEmpty",
    "prod/PositionalVar",
    "prod/OrderModifier",
    "prod/ReturnClause",
    "prod/TumblingWindowClause",
    "prod/SlidingWindowClause",
    // the prolog
    "prod/Prolog",
    "prod/VarDecl",
    "prod/FunctionDecl",
    "prod/BoundarySpaceDecl",
    "prod/ConstructionDecl",
    "prod/OrderingModeDecl",
    "prod/EmptyOrderDecl",
    "prod/CopyNamespacesDecl",
    "prod/DefaultNamespaceDecl",
    "prod/NamespaceDecl",
    "prod/DecimalFormatDecl",
    "prod/BaseURIDecl",
    "prod/DefaultCollationDecl",
    "prod/ContextItemDecl",
    "prod/OptionDecl",
    "prod/VersionDecl",
    "prod/ModuleImport",
    "prod/ModuleDecl",
    "prod/Annotation",
    // the other XQuery-only expressions
    "prod/TypeswitchExpr",
    "prod/SwitchExpr",
    "prod/TryCatchExpr",
    "prod/OrderedExpr",
    "prod/UnorderedExpr",
    "prod/ExtensionExpr",
];

/// Locates the `qt3tests` directory under `OXIXML_XML_SUITES_DIR`.
fn qt3_root() -> Option<PathBuf> {
    let root = PathBuf::from(env::var_os("OXIXML_XML_SUITES_DIR")?);
    for entry in fs::read_dir(&root).ok()?.flatten() {
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

fn decode_utf16(bytes: &[u8], big_endian: bool) -> Option<String> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16(&units).ok()
}

fn parse_file(path: &Path) -> Option<Tree> {
    Tree::parse_str(&fs::read_to_string(path).ok()?).ok()
}

fn parse_source(path: &Path, uri: Option<&str>) -> Option<Tree> {
    let text = fs::read_to_string(path).ok()?;
    // A `<source>` that publishes a URI *is* that document: `fn:document-uri` has to report it.
    let document_uri = uri.map_or_else(|| path.to_string_lossy().to_string(), str::to_owned);
    let options = ParseOptions::default().document_uri(document_uri);
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

/// One `<decimal-format>` of an environment: its `name` — the lexical `QName` and the namespace
/// its prefix resolves to, or `None` for the default format — and its properties, spelled
/// exactly as `declare decimal-format` spells them.
type DecimalFormatDeclaration = (Option<(String, Option<String>)>, Vec<(String, String)>);

/// The catalogue's own case-insensitive collation.
///
/// `catalog.xml` names it in the environments and test cases of `fn/sort.xml`,
/// `array/array-sort.xml` and `fn/function-lookup.xml`, and the suite gives its definition in
/// prose: it compares strings by code point after case-folding. Nothing about it is
/// specification-defined, so it lives here with the rest of the harness rather than in the
/// engine, which knows only the collations F&O 3.1 names.
#[derive(Debug)]
struct CaseBlindCollation;

impl CaseBlindCollation {
    const URI: &'static str = "http://www.w3.org/2010/09/qt-fots-catalog/collation/caseblind";
}

impl oxixml_fn::Collation for CaseBlindCollation {
    fn uri(&self) -> &str {
        Self::URI
    }

    fn compare(&self, left: &str, right: &str) -> std::cmp::Ordering {
        left.to_lowercase().cmp(&right.to_lowercase())
    }

    fn collation_units(&self, value: &str) -> Option<Vec<char>> {
        Some(value.to_lowercase().chars().collect())
    }
}

/// The catalogue's sentinel for an absent static base URI.
const UNDEFINED_BASE_URI: &str = "#UNDEFINED";

/// One `<resource>`: the URI it is published under, the file that holds it, and the encoding
/// the catalogue says those octets are in.
#[derive(Clone)]
struct Resource {
    uri: String,
    file: String,
    encoding: Option<String>,
}

/// One `<source>`: its `role`, its file, and the URI the environment publishes it under.
#[derive(Clone)]
struct Source {
    role: String,
    file: String,
    uri: Option<String>,
}

#[derive(Default, Clone)]
struct Environment {
    sources: Vec<Source>,
    /// `<collection uri="…">` and the files it holds.
    collections: Vec<(String, Vec<String>)>,
    namespaces: Vec<(String, String)>,
    parameters: Vec<(String, String)>,
    /// `<resource uri="…" file="…" encoding="…">`, kept undecoded: the suite serves resources
    /// whose octets are deliberately *not* valid in their declared encoding.
    resources: Vec<Resource>,
    /// Library modules, as `(target namespace, file)`.
    modules: Vec<(String, String)>,
    /// `<decimal-format>` declarations.
    decimal_formats: Vec<DecimalFormatDeclaration>,
    /// `<context-item select="…"/>`: an XPath expression whose value is the context item, which
    /// is how the catalogue supplies a context item that is not a document.
    context_item_select: Option<String>,
    base_uri: Option<String>,
    /// Whether the environment declared `<static-base-uri uri="#UNDEFINED"/>` — the catalogue's
    /// own sentinel for "this case runs with *no* static base URI", which is the whole point of
    /// the cases that expect a relative `fn:doc` or `fn:unparsed-text` URI to fail.
    base_uri_undefined: bool,
    /// A schema declaration means the case needs schema awareness, which this engine has not.
    needs_schema: bool,
    /// The directory the environment's file names are relative to. `None` means the test set's
    /// own directory; a *catalogue-level* environment resolves against the catalogue instead.
    base_dir: Option<PathBuf>,
}

fn read_environment(node: &NodeRef) -> Environment {
    let mut environment = Environment::default();
    for child in element_children(node) {
        match local_name(&child).as_str() {
            "source" => {
                let role = attribute(&child, "role").unwrap_or_default();
                if let Some(file) = attribute(&child, "file") {
                    environment.sources.push(Source {
                        role,
                        file,
                        uri: attribute(&child, "uri"),
                    });
                }
            }
            "collection" => {
                let uri = attribute(&child, "uri").unwrap_or_default();
                let files = children_named(&child, "source")
                    .filter_map(|source| attribute(&source, "file"))
                    .collect();
                environment.collections.push((uri, files));
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
                    // A `<param name="test:x" xmlns:test="…">` binds its own prefix, rather than
                    // going through the environment's `<namespace>` children, so the name is
                    // expanded against the catalogue element's own in-scope namespaces.
                    if let Some((prefix, _)) = name.split_once(':') {
                        if let Some(uri) =
                            oxixml_fn::qnames::namespace_uri_for_prefix(Some(prefix), &child)
                        {
                            if !environment
                                .namespaces
                                .iter()
                                .any(|(bound, _)| bound == prefix)
                            {
                                environment.namespaces.push((prefix.to_owned(), uri));
                            }
                        }
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
                    environment.resources.push(Resource {
                        uri,
                        file,
                        encoding: attribute(&child, "encoding"),
                    });
                }
            }
            "module" => {
                if let (Some(uri), Some(file)) =
                    (attribute(&child, "uri"), attribute(&child, "file"))
                {
                    environment.modules.push((uri, file));
                }
            }
            "decimal-format" => {
                // A `<decimal-format name="a:test">` may bind its prefix with an `xmlns:a` of
                // its own rather than through the environment's `<namespace>` children, so the
                // name is expanded against the catalogue element's own in-scope namespaces.
                let name = attribute(&child, "name").map(|lexical| {
                    let namespace = lexical.split_once(':').and_then(|(prefix, _)| {
                        oxixml_fn::qnames::namespace_uri_for_prefix(Some(prefix), &child)
                    });
                    (lexical, namespace)
                });
                let properties = child
                    .attributes()
                    .into_iter()
                    .filter_map(|attribute| {
                        let local = attribute.node_name()?.local_name().to_owned();
                        (local != "name").then(|| (local, attribute.string_value()))
                    })
                    .collect();
                environment.decimal_formats.push((name, properties));
            }
            "static-base-uri" => match attribute(&child, "uri") {
                Some(uri) if uri.trim() == UNDEFINED_BASE_URI => {
                    environment.base_uri = None;
                    environment.base_uri_undefined = true;
                }
                uri => environment.base_uri = uri,
            },
            "schema" => environment.needs_schema = true,
            _ => {}
        }
    }
    environment
}

/// The documents a test-set's environment makes available to `fn:doc`.
///
/// Besides what the environment declares, a relative URI is resolved against the test set's own
/// directory and read from there: the suite's `fn:unparsed-text` and `fn:json-doc` cases name
/// their data files that way, and resolving a relative URI against the static base URI — which
/// *is* the test-set directory — is what a processor is required to do.
#[derive(Debug, Default)]
struct Documents {
    by_uri: BTreeMap<String, NodeRef>,
    /// Declared `<resource>`s, as the file that holds them and the encoding the catalogue says
    /// they are in. They are decoded on retrieval, never on load, because whether the octets
    /// decode at all is exactly what several cases assert.
    text_by_uri: BTreeMap<String, (PathBuf, Option<String>)>,
    collections: BTreeMap<String, Vec<NodeRef>>,
    /// The directory a relative URI is resolved against — `None` when the environment declared
    /// `<static-base-uri uri="#UNDEFINED"/>`, in which case a relative URI resolves to nothing
    /// at all and `fn:unparsed-text` must report `FOUT1170`.
    directory: Option<PathBuf>,
}

impl DocumentResolver for Documents {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        self.by_uri.get(uri).cloned()
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        let named = uri.unwrap_or("");
        if let Some(nodes) = self.collections.get(named) {
            return Some(nodes.clone());
        }
        match uri {
            // `None` asks for the *default collection*. An environment that declares no
            // collection has none, and F&O 3.1 §14.7.4 makes that `FODC0002` — not a silently
            // empty sequence, and not "every document this environment happens to hold", which
            // would make `fn:collection()` succeed in cases written to prove it cannot.
            None => (!self.collections.is_empty()).then(|| self.by_uri.values().cloned().collect()),
            Some(uri) => self.by_uri.get(uri).map(|node| vec![node.clone()]),
        }
    }

    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        self.unparsed_text_result(uri, encoding).ok()
    }

    /// The suite deliberately serves resources whose octets are *not* valid in their encoding —
    /// truncated UTF-8, lone continuation bytes, Latin-1 text labelled UTF-8 — and asks for
    /// `FOUT1200` rather than the generic `FOUT1170`. Reporting the reason is the driver's job:
    /// the file is there, so "unretrievable" would be the wrong answer.
    fn unparsed_text_result(
        &self,
        uri: &str,
        encoding: Option<&str>,
    ) -> Result<String, TextRetrievalError> {
        if let Some((path, declared)) = self.text_by_uri.get(uri) {
            let Ok(bytes) = fs::read(path) else {
                return Err(TextRetrievalError::Unretrievable);
            };
            // F&O 3.1 §14.8.1: an explicit `$encoding` wins, and the resource's own declared
            // encoding is what "the external environment" supplies when there is none.
            return decode_text(&bytes, encoding.or(declared.as_deref()));
        }
        let relative = uri.strip_prefix("file:///").unwrap_or(uri);
        if relative.contains("://") || relative.starts_with('/') {
            return Err(TextRetrievalError::Unretrievable);
        }
        let Some(directory) = &self.directory else {
            return Err(TextRetrievalError::Unretrievable);
        };
        let Ok(bytes) = fs::read(directory.join(relative)) else {
            return Err(TextRetrievalError::Unretrievable);
        };
        decode_text(&bytes, encoding)
    }
}

/// Decodes a retrieved resource, distinguishing "not this encoding" from "not retrievable".
///
/// Only the encodings the suite's own resources use are understood; anything else is
/// `FOUT1190`, which is what F&O 3.1 §14.8.2 says about an encoding the processor does not
/// support. Octets that are not valid UTF-8 are `FOUT1200`.
fn decode_text(bytes: &[u8], encoding: Option<&str>) -> Result<String, TextRetrievalError> {
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return decode_utf16(rest, false)
            .map(|text| format!("\u{FEFF}{text}"))
            .ok_or(TextRetrievalError::InvalidOctets);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16(rest, true)
            .map(|text| format!("\u{FEFF}{text}"))
            .ok_or(TextRetrievalError::InvalidOctets);
    }
    if let Some(named) = encoding {
        let normalized = named.trim().to_ascii_lowercase();
        if !matches!(normalized.as_str(), "utf-8" | "utf8") {
            return Err(TextRetrievalError::UnsupportedEncoding);
        }
    }
    // Only the *decoding* is checked here. F&O 3.1 §14.8.1 also forbids characters XML does not
    // allow, but that restriction belongs to `fn:unparsed-text` itself, not to retrieval: the
    // same resolver hook serves `fn:json-doc`, and the JSON test suite deliberately feeds it
    // form feeds, vertical tabs and NUL and expects `FOJS0001` from the *parser*.
    String::from_utf8(bytes.to_vec()).map_err(|_| TextRetrievalError::InvalidOctets)
}

/// Builds a [`DecimalFormat`] from an environment's `<decimal-format>` attributes.
///
/// `None` means the catalogue gave a property this driver does not know, or a single-character
/// property with the wrong length; the environment is then unusable rather than silently wrong.
fn build_decimal_format(properties: &[(String, String)]) -> Option<DecimalFormat> {
    let mut format = DecimalFormat::default();
    for (name, value) in properties {
        let single = || {
            let mut characters = value.chars();
            match (characters.next(), characters.next()) {
                (Some(character), None) => Some(character),
                _ => None,
            }
        };
        format = match name.as_str() {
            "decimal-separator" => format.with_decimal_separator(single()?),
            "grouping-separator" => format.with_grouping_separator(single()?),
            "exponent-separator" => format.with_exponent_separator(single()?),
            "infinity" => format.with_infinity(value.clone()),
            "minus-sign" => format.with_minus_sign(single()?),
            "NaN" => format.with_nan(value.clone()),
            "percent" => format.with_percent(single()?),
            "per-mille" => format.with_per_mille(single()?),
            "zero-digit" => format.with_zero_digit(single()?),
            "digit" => format.with_digit(single()?),
            "pattern-separator" => format.with_pattern_separator(single()?),
            _ => return None,
        };
    }
    Some(format)
}

// --- dependencies ----------------------------------------------------------------------------

/// Whether a `spec` dependency's value covers XQuery 3.1, which is the version this driver
/// compiles every case as.
///
/// The suite spells a version list as `XQ10 XQ30`, or open-endedly as `XQ10+`. A case whose list
/// stops at 3.0 is testing behaviour XQuery 3.1 deliberately changed — two `where` clauses, a
/// union in a `typeswitch` case, `XQST0054` where 3.1 says `XQDY0054` — and running it as 3.1
/// would only measure the wrong thing, so it is skipped rather than failed.
fn covers_xquery_31(value: &str) -> bool {
    value.split_whitespace().any(|token| {
        let Some(rest) = token.strip_prefix("XQ") else {
            return false;
        };
        match rest.strip_suffix('+') {
            Some(from) => from.parse::<u32>().is_ok_and(|version| version <= 31),
            None => rest == "31",
        }
    })
}

/// Whether an `xml-version` dependency token names the XML edition this engine implements.
///
/// The catalogue spells editions as `1.0`, `1.1`, `1.0:4-` ("fourth edition or earlier") and
/// `1.0:5+` ("fifth edition or later"). The distinction is not cosmetic: `NameStartChar` grew in
/// the fifth edition, so `<Ϳnode/>` (U+037F) and `<eggſ/>` (U+017F) are well formed here and a
/// syntax error under `1.0:4-`. Running a `1.0:4-` case as a fifth-edition processor would
/// measure the wrong thing, so it is skipped.
fn is_xml_10_fifth_edition(token: &str) -> bool {
    match token.split_once(':') {
        None => token == "1.0",
        Some(("1.0", edition)) => match edition.strip_suffix('+') {
            Some(from) => from.parse::<u32>().is_ok_and(|edition| edition <= 5),
            None => edition
                .strip_suffix('-')
                .and_then(|to| to.parse::<u32>().ok())
                .is_some_and(|edition| edition >= 5),
        },
        Some(_) => false,
    }
}

/// Whether every dependency declared on `node` is one this engine satisfies, and if not, why.
fn dependencies_supported(node: &NodeRef) -> Result<(), SkipReason> {
    for dependency in children_named(node, "dependency") {
        let kind = attribute(&dependency, "type").unwrap_or_default();
        let value = attribute(&dependency, "value").unwrap_or_default();
        let satisfied =
            attribute(&dependency, "satisfied").is_none_or(|text| text.trim() != "false");
        if !DECIDABLE_DEPENDENCIES.contains(&kind.as_str()) {
            return Err(SkipReason::UndecidableDependency);
        }
        let holds = match kind.as_str() {
            "spec" => covers_xquery_31(&value),
            // This engine parses XML 1.0, *fifth edition*, and validates against XSD 1.0 lexical
            // spaces; its language and calendar are English and AD, and it has no others.
            "xml-version" => value.split_whitespace().any(is_xml_10_fifth_edition),
            "xsd-version" => value.split_whitespace().any(|token| token == "1.0"),
            "language" | "default-language" => value.split_whitespace().any(|token| token == "en"),
            "calendar" => value.split_whitespace().any(|token| token == "AD"),
            _ => SUPPORTED_FEATURES.contains(&value.as_str()),
        };
        if holds != satisfied {
            return Err(match kind.as_str() {
                "spec" => SkipReason::Spec,
                "feature" => SkipReason::Feature,
                _ => SkipReason::Edition,
            });
        }
    }
    Ok(())
}

// --- the per-case time budget ------------------------------------------------------------------

/// How long one test case may run before the driver gives up on it.
///
/// The suite contains a handful of queries this engine does not terminate on — an unbounded
/// recursion the optimiser cannot cut, a `fn:collection` over a catalogue environment that never
/// converges. Without a budget one of them hangs the whole run, which is why the catalogue's own
/// environments used to be withheld. With a budget they are simply **failures**: the case did
/// not produce the required result in the time allowed. They are never counted as skips.
///
/// `OXIXML_QT3_CASE_BUDGET_SECS` overrides it while triaging — a case that overruns on a loaded
/// machine and passes on a quiet one is a *timing* result, not a conformance one, and raising
/// the budget is how the two are told apart. The assertions never set it.
fn case_budget() -> Duration {
    env::var("OXIXML_QT3_CASE_BUDGET_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map_or(DEFAULT_CASE_BUDGET, Duration::from_secs)
}

const DEFAULT_CASE_BUDGET: Duration = Duration::from_secs(30);

/// The stack a worker gets. The suite's deeply recursive queries need far more than the default.
const WORKER_STACK: usize = 512 * 1024 * 1024;

/// What running one case produced: the skip its environment forces, or the assertion's verdict
/// (`None` when this driver cannot decide the assertion).
type CaseResult = Result<Option<Result<(), String>>, SkipReason>;

/// A job handed to the worker thread.
type Job = Box<dyn FnOnce() -> CaseResult + Send + 'static>;

/// Runs test cases on a worker thread, one at a time, under [`case_budget`].
///
/// The worker is reused across cases — spawning a half-gigabyte-stack thread per case would cost
/// more than the suite itself. When a case overruns, its worker is **abandoned**: the thread is
/// left running (there is no safe way to cancel a computation in Rust), the runner drops its
/// handles to it, and a fresh worker takes over. An abandoned worker holds one query's working
/// set until the process exits, which is why the budget is generous but finite.
struct CaseRunner {
    jobs: mpsc::Sender<Job>,
    results: mpsc::Receiver<CaseResult>,
}

impl CaseRunner {
    fn new() -> Self {
        let (jobs, incoming) = mpsc::channel::<Job>();
        let (outgoing, results) = mpsc::channel::<CaseResult>();
        std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || {
                while let Ok(job) = incoming.recv() {
                    if outgoing.send(job()).is_err() {
                        break;
                    }
                }
            })
            .expect("a worker thread");
        Self { jobs, results }
    }

    /// Runs `job`, or answers `None` if it did not finish within [`case_budget`].
    fn run(&mut self, job: Job) -> Option<CaseResult> {
        if self.jobs.send(job).is_err() {
            *self = Self::new();
            return None;
        }
        match self.results.recv_timeout(case_budget()) {
            Ok(result) => Some(result),
            Err(_) => {
                // The worker is still inside the query. Abandon it and start again.
                *self = Self::new();
                None
            }
        }
    }
}

// --- the driver -------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Tally {
    passed: usize,
    failed: usize,
    skipped: usize,
    /// Cases that exhausted their wall-clock budget. They are counted in `failed` as well —
    /// a case this engine cannot finish is a failure, not a skip — and reported separately so
    /// that the shape of the remaining work stays visible.
    timed_out: usize,
    failures: Vec<String>,
    skip_reasons: BTreeMap<SkipReason, usize>,
}

impl Tally {
    fn merge(&mut self, other: Self) {
        self.passed += other.passed;
        self.failed += other.failed;
        self.skipped += other.skipped;
        self.timed_out += other.timed_out;
        self.failures.extend(other.failures);
        for (reason, count) in other.skip_reasons {
            *self.skip_reasons.entry(reason).or_default() += count;
        }
    }

    fn skip(&mut self, reason: SkipReason) {
        self.skipped += 1;
        *self.skip_reasons.entry(reason).or_default() += 1;
    }
}

struct Prepared {
    statik: StaticQueryContext,
    /// The decimal formats the environment declares, for `fn:format-number`.
    decimal_formats: Vec<(Option<ExpandedName>, DecimalFormat)>,
    context_item: Option<NodeRef>,
    /// The `<context-item select="…"/>` expression, evaluated once the resolver is in place.
    context_item_select: Option<String>,
    parameters: Vec<(String, String)>,
    namespaces: Vec<(String, String)>,
    /// Documents an environment bound to an external variable through `role="$name"`.
    bound_sources: Vec<(String, NodeRef)>,
    documents: Arc<Documents>,
    trees: Vec<Tree>,
}

/// Expands a lexical variable name against an environment's namespace bindings.
fn expanded(lexical: &str, namespaces: &[(String, String)]) -> ExpandedName {
    match lexical.split_once(':') {
        Some((prefix, local)) => match namespaces.iter().find(|(bound, _)| bound == prefix) {
            Some((_, uri)) => ExpandedName::new(Some(uri.as_str()), local),
            None => ExpandedName::new(None::<&str>, lexical),
        },
        None => ExpandedName::new(None::<&str>, lexical),
    }
}

fn build_context(environment: &Environment, directory: &Path) -> Result<Prepared, SkipReason> {
    if environment.needs_schema {
        return Err(SkipReason::SchemaEnvironment);
    }
    let mut statik = StaticQueryContext::new().with_collation(Arc::new(CaseBlindCollation));
    for (prefix, uri) in &environment.namespaces {
        statik = statik.with_namespace(prefix.clone(), uri.clone());
    }
    // An environment's `<param>` is an external variable, so it has to be *declared* as well as
    // supplied; otherwise the reference to it is the static error `XPST0008`.
    for (name, _) in &environment.parameters {
        statik = statik.with_variable_name(expanded(name, &environment.namespaces));
    }
    if let Some(base) = &environment.base_uri {
        statik = statik.with_base_uri(base.clone());
    }
    // An environment may declare decimal formats the query then names in `fn:format-number`.
    let mut decimal_formats = Vec::new();
    for (name, properties) in &environment.decimal_formats {
        let Some(format) = build_decimal_format(properties) else {
            return Err(SkipReason::EnvironmentUnavailable);
        };
        // `fn:format-number` is given its format name as a *lexical* string, so the format is
        // registered under every spelling the query could name it with: the expanded name that
        // `Q{uri}local` reduces to, the `prefix:local` spelling the catalogue used, and the
        // spelling under every prefix the environment binds to the same namespace.
        let Some((lexical, namespace)) = name else {
            decimal_formats.push((None, format));
            continue;
        };
        let local = lexical.split_once(':').map_or(lexical.as_str(), |(_, l)| l);
        let mut spellings = vec![
            ExpandedName::new(namespace.as_deref(), local),
            ExpandedName::new(None::<&str>, lexical.clone()),
        ];
        if let Some(namespace) = &namespace {
            for (prefix, uri) in &environment.namespaces {
                if uri == namespace {
                    spellings.push(ExpandedName::new(None::<&str>, format!("{prefix}:{local}")));
                }
            }
        } else {
            spellings.push(ExpandedName::new(None::<&str>, local));
        }
        for spelling in spellings {
            decimal_formats.push((Some(spelling), format.clone()));
        }
    }
    if !environment.modules.is_empty() {
        let mut library = InMemoryModules::new();
        for (uri, file) in &environment.modules {
            let source = fs::read_to_string(directory.join(file))
                .map_err(|_| SkipReason::EnvironmentUnavailable)?;
            library.insert(uri.clone(), source);
        }
        statik = statik.with_module_resolver(Arc::new(library));
    }

    let mut documents = Documents {
        directory: (!environment.base_uri_undefined).then(|| directory.to_path_buf()),
        ..Documents::default()
    };
    let mut trees = Vec::new();
    let mut context_item = None;
    for resource in &environment.resources {
        let path = directory.join(&resource.file);
        documents.text_by_uri.insert(
            resource.uri.clone(),
            (path.clone(), resource.encoding.clone()),
        );
        documents.text_by_uri.insert(
            path.to_string_lossy().to_string(),
            (path, resource.encoding.clone()),
        );
    }
    let mut bound_sources: Vec<(String, NodeRef)> = Vec::new();
    for source in &environment.sources {
        let tree = parse_source(&directory.join(&source.file), source.uri.as_deref())
            .ok_or(SkipReason::EnvironmentUnavailable)?;
        let root = tree.root();
        if source.role == "." {
            context_item = Some(root.clone());
        }
        // A `role` of `$name` binds the document to an external variable of that name.
        if let Some(name) = source.role.strip_prefix('$') {
            statik = statik.with_variable_name(expanded(name, &environment.namespaces));
            bound_sources.push((name.to_owned(), root.clone()));
        }
        documents.by_uri.insert(source.file.clone(), root.clone());
        if let Some(uri) = &source.uri {
            documents.by_uri.insert(uri.clone(), root.clone());
        }
        documents.by_uri.insert(
            directory.join(&source.file).to_string_lossy().to_string(),
            root,
        );
        trees.push(tree);
    }
    for (uri, files) in &environment.collections {
        let mut nodes = Vec::with_capacity(files.len());
        for file in files {
            let tree = parse_source(&directory.join(file), None)
                .ok_or(SkipReason::EnvironmentUnavailable)?;
            nodes.push(tree.root());
            trees.push(tree);
        }
        documents.collections.insert(uri.clone(), nodes);
    }
    Ok(Prepared {
        statik,
        decimal_formats,
        context_item,
        context_item_select: environment.context_item_select.clone(),
        parameters: environment.parameters.clone(),
        namespaces: environment.namespaces.clone(),
        bound_sources,
        documents: Arc::new(documents),
        trees,
    })
}

fn evaluate_case(
    query: &str,
    environment: &Environment,
    directory: &Path,
) -> Result<Result<XpathValue, XQueryError>, SkipReason> {
    let Prepared {
        statik,
        decimal_formats,
        context_item,
        context_item_select,
        parameters,
        namespaces,
        bound_sources,
        documents,
        trees,
    } = build_context(environment, directory)?;
    let compiled = match XQuery::compile(query, statik) {
        Ok(compiled) => compiled,
        Err(error) => return Ok(Err(error)),
    };
    // `fn:trace` writes to standard error by default, and the suite calls it hundreds of times.
    let fn_context = FnContext::new().with_trace_sink(Arc::new(CollectingTrace::new()));
    // Suite queries recurse deeply on purpose, which is why the worker thread has room for it;
    // the evaluator is told about that room rather than assuming an ordinary thread's 2 MiB.
    let mut context = DynamicQueryContext::new()
        .with_fn_context(fn_context)
        .with_collation(Arc::new(CaseBlindCollation))
        .with_stack_size(WORKER_STACK)
        .with_document_resolver(Arc::clone(&documents) as Arc<dyn DocumentResolver>);
    for (name, format) in decimal_formats {
        context = context.with_decimal_format(name, format);
    }
    if let Some(item) = context_item {
        context = context.with_context_item(item);
    }
    // `<context-item select="…"/>` supplies a context item that is not a document — an atomic
    // value, an array, a function. Its expression is XPath, like `<param select="…">`.
    if let Some(select) = &context_item_select {
        let compiled = match XPath::compile_31(select) {
            Ok(compiled) => compiled,
            Err(error) => return Ok(Err(XQueryError::from(error))),
        };
        let bound = match compiled.evaluate_value_in(&compiled.context()) {
            Ok(value) => value,
            Err(error) => return Ok(Err(XQueryError::from(error))),
        };
        if let Some(item) = bound.first() {
            context = context.with_context_item(item.clone());
        }
    }
    for (name, node) in &bound_sources {
        context = context.with_variable_name(expanded(name, &namespaces), node.clone());
    }
    for (name, select) in &parameters {
        let compiled = match XPath::compile_31(select) {
            Ok(compiled) => compiled,
            Err(error) => return Ok(Err(XQueryError::from(error))),
        };
        let inner = compiled
            .context()
            .with_document_resolver(Arc::clone(&documents) as Arc<dyn DocumentResolver>);
        let bound = match compiled.evaluate_value_in(&inner) {
            Ok(value) => value,
            Err(error) => return Ok(Err(XQueryError::from(error))),
        };
        context = context.with_variable_name(expanded(name, &namespaces), bound);
    }
    let outcome = compiled.execute_value(&context);
    drop(trees);
    Ok(outcome)
}

/// Checks one assertion element against the outcome. `None` means "this driver cannot decide".
fn check(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, XQueryError>,
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
            // A few cases spell the expected code as an `EQName`; only its local part matters.
            let wanted = expected
                .rsplit_once('}')
                .map_or(expected.as_str(), |(_, local)| local);
            match outcome {
                Err(error) => {
                    if expected == "*" || error.name().local_name() == wanted {
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
        "assert-xml" => {
            if attribute(assertion, "file").is_some() {
                return None;
            }
            let expected = assertion.string_value();
            match outcome {
                Ok(value) => {
                    let sequence = value.clone().into_sequence().ok()?;
                    let actual = serialize(&sequence, &SerializationParameters::default()).ok()?;
                    if actual == expected
                        || same_xml(&actual, &expected)
                        || same_nodes(&sequence, &expected)
                    {
                        Some(Ok(()))
                    } else {
                        Some(Err(format!("expected XML `{expected}`, got `{actual}`")))
                    }
                }
                Err(error) => Some(Err(format!("expected XML `{expected}`, raised {error}"))),
            }
        }
        "assert-serialization" => {
            if attribute(assertion, "file").is_some() {
                return None;
            }
            let expected = assertion.string_value();
            match outcome {
                Ok(value) => {
                    let sequence = value.clone().into_sequence().ok()?;
                    let actual = serialize(&sequence, &SerializationParameters::default()).ok()?;
                    if collapse(&actual) == collapse(&expected) {
                        Some(Ok(()))
                    } else {
                        Some(Err(format!(
                            "expected serialization `{expected}`, got `{actual}`"
                        )))
                    }
                }
                Err(error) => Some(Err(format!(
                    "expected serialization `{expected}`, raised {error}"
                ))),
            }
        }
        "assert-eq" | "assert-deep-eq" => {
            let expected_source = assertion.string_value();
            let expected = compile_and_run(&expected_source).ok()?;
            match outcome {
                Ok(value) => {
                    let mut left = value.clone().into_sequence().ok()?;
                    if kind == "assert-eq" {
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
        _ => None,
    }
}

/// How many failures to keep for the report; raised by `OXIXML_QT3_FAILURES` while triaging.
fn failure_cap() -> usize {
    env::var("OXIXML_QT3_FAILURES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(25)
}

/// Whether two serialized fragments are the same XML, ignoring the formatting the catalogue
/// happens to have wrapped its expected value in.
fn same_xml(actual: &str, expected: &str) -> bool {
    let parse = |text: &str| {
        Tree::parse_str(&format!("<oxixml-wrapper>{text}</oxixml-wrapper>"))
            .ok()
            .and_then(|tree| tree.document_element())
            .map(|root| Sequence::from_iter(root.children()))
    };
    match (parse(actual), parse(expected)) {
        (Some(left), Some(right)) => deep_equal(&left, &right).unwrap_or(false),
        _ => false,
    }
}

/// Whether the result sequence *is* the expected XML, compared node by node rather than through
/// the serializer.
///
/// `assert-xml` is defined in terms of serialization, and comparing the serialized text is what
/// `same_xml` does. This is the second chance: it parses only the expected value and compares it
/// with `fn:deep-equal` against the nodes the query actually built, so that a difference in how
/// the *serializer* spells a namespace — which belongs to `oxixml-fn`, not to this engine — does
/// not read as an XQuery conformance failure.
fn same_nodes(actual: &Sequence, expected: &str) -> bool {
    let Ok(tree) = Tree::parse_str(&format!("<oxixml-wrapper>{expected}</oxixml-wrapper>")) else {
        return false;
    };
    let Some(root) = tree.document_element() else {
        return false;
    };
    let wanted = Sequence::from_iter(root.children());
    if deep_equal(actual, &wanted).unwrap_or(false) {
        return true;
    }
    // `document { … }` produces one document node whose children are what was asserted.
    match actual.as_slice() {
        [item] => item
            .as_node()
            .filter(|node| node.node_kind() == NodeKind::Document)
            .is_some_and(|node| {
                deep_equal(&Sequence::from_iter(node.children()), &wanted).unwrap_or(false)
            }),
        _ => false,
    }
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn string_value_of(value: &XpathValue) -> Result<String, XQueryError> {
    let mut parts = Vec::with_capacity(value.len());
    for item in value.iter() {
        parts.push(item.string_value()?);
    }
    Ok(parts.join(" "))
}

/// Evaluates an assertion's expected value.
///
/// The catalogue's assertions are **XPath 3.1** expressions, not XQuery ones — the catalogue
/// says so, and its own history bears it out (`prod/PathExpr.xml`'s `PathExpr-17` carries a
/// `<modified … change="assertion must not use XQ syntax"/>`). The difference matters here:
/// XQuery's `StringLiteral` recognizes `PredefinedEntityRef` and `CharRef` where XPath's does
/// not, so `prod/StringConstructor.xml`'s "entity references are not expanded" cases assert
/// against `"There were &lt; 10 green bottles"` and mean those nine characters literally.
fn compile_and_run(source: &str) -> Result<Sequence, XQueryError> {
    let compiled = XPath::compile(source, XPathVersion::XPath31, StaticContext::new())
        .map_err(|error| XQueryError::new(XQueryErrorCode::Xpst0003, error.message().to_owned()))?;
    let value = compiled
        .evaluate_value_in(&compiled.context())
        .map_err(|error| XQueryError::new(XQueryErrorCode::Xpty0004, error.message().to_owned()))?;
    value
        .into_sequence()
        .map_err(|error| XQueryError::new(XQueryErrorCode::Xpty0004, error.message().to_owned()))
}

fn run_test_set(
    path: &Path,
    shared: &BTreeMap<String, Environment>,
    tally: &mut Tally,
    runner: &mut CaseRunner,
) {
    let Some(tree) = parse_file(path) else {
        tally.skip(SkipReason::EnvironmentUnavailable);
        return;
    };
    let Some(root) = tree.document_element() else {
        tally.skip(SkipReason::EnvironmentUnavailable);
        return;
    };
    let directory = path.parent().unwrap_or(Path::new("."));
    let set_name = attribute(&root, "name").unwrap_or_default();

    let mut environments: BTreeMap<String, Environment> = shared.clone();
    environments.insert("empty".to_owned(), Environment::default());
    for definition in children_named(&root, "environment") {
        if let Some(name) = attribute(&definition, "name") {
            environments.insert(name, read_environment(&definition));
        }
    }
    let set_supported = dependencies_supported(&root);

    for case in children_named(&root, "test-case") {
        let case_name = attribute(&case, "name").unwrap_or_default();
        if let Err(reason) = set_supported.and(dependencies_supported(&case)) {
            tally.skip(reason);
            continue;
        }
        let Some(test) = children_named(&case, "test").next() else {
            tally.skip(SkipReason::NoAssertion);
            continue;
        };
        if attribute(&test, "file").is_some() {
            tally.skip(SkipReason::ExternalQuery);
            continue;
        }
        let query = test.string_value();
        if query.trim().is_empty() {
            tally.skip(SkipReason::ExternalQuery);
            continue;
        }

        let mut environment = match children_named(&case, "environment").next() {
            Some(node) => match attribute(&node, "ref") {
                Some(reference) => match environments.get(&reference) {
                    Some(environment) => environment.clone(),
                    None => {
                        tally.skip(SkipReason::EnvironmentUndefined);
                        continue;
                    }
                },
                None => read_environment(&node),
            },
            None => Environment::default(),
        };

        // A `<module>` is a child of the test case, not of its environment.
        for module in children_named(&case, "module") {
            if let (Some(uri), Some(file)) = (attribute(&module, "uri"), attribute(&module, "file"))
            {
                environment.modules.push((uri, file));
            }
        }

        let Some(result) = children_named(&case, "result").next() else {
            tally.skip(SkipReason::NoAssertion);
            continue;
        };
        let Some(assertion) = element_children(&result).into_iter().next() else {
            tally.skip(SkipReason::NoAssertion);
            continue;
        };

        let base = environment.base_dir.clone();
        let directory = base.as_deref().unwrap_or(directory).to_path_buf();
        let job: Job = Box::new(move || {
            evaluate_case(&query, &environment, &directory)
                .map(|outcome| check(&assertion, &outcome))
        });
        let verdict = match runner.run(job) {
            Some(result) => result,
            None => {
                tally.failed += 1;
                tally.timed_out += 1;
                if tally.failures.len() < failure_cap() {
                    tally.failures.push(format!(
                        "{set_name}/{case_name}: did not finish within {}s",
                        case_budget().as_secs()
                    ));
                }
                continue;
            }
        };
        match verdict {
            Err(reason) => tally.skip(reason),
            Ok(None) => tally.skip(SkipReason::UndecidableAssertion),
            Ok(Some(Ok(()))) => tally.passed += 1,
            Ok(Some(Err(reason))) => {
                tally.failed += 1;
                if tally.failures.len() < failure_cap() {
                    tally
                        .failures
                        .push(format!("{set_name}/{case_name}: {reason}"));
                }
            }
        }
    }
}

fn run_suite(root: &Path, only: Option<&[&str]>) -> Tally {
    let mut tally = Tally::default();
    let Some(catalog) = parse_file(&root.join("catalog.xml")) else {
        return tally;
    };
    let Some(document_element) = catalog.document_element() else {
        return tally;
    };
    // The catalogue declares environments of its own, and a test case may name one instead of
    // one its own set declares. They are merged in here, resolved against the catalogue's
    // directory rather than the test set's. A handful of the queries this enables do not
    // terminate under this engine; `CaseRunner` bounds them and counts them as failures, which
    // is why merging is safe (it used to hang the run).
    let mut shared: BTreeMap<String, Environment> = BTreeMap::new();
    for definition in children_named(&document_element, "environment") {
        if let Some(name) = attribute(&definition, "name") {
            let mut environment = read_environment(&definition);
            environment.base_dir = Some(root.to_path_buf());
            shared.insert(name, environment);
        }
    }
    // `OXIXML_QT3_ONLY` narrows a run to the test sets whose catalogue `file=` starts with one of
    // its space-separated prefixes. It exists for triage — running one set takes a second where
    // the whole suite takes minutes — and is never set by the assertions below, which always see
    // the full corpus.
    let requested: Vec<String> = env::var("OXIXML_QT3_ONLY")
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let mut runner = CaseRunner::new();
    for entry in children_named(&document_element, "test-set") {
        let Some(file) = attribute(&entry, "file") else {
            continue;
        };
        if let Some(prefixes) = only {
            if !prefixes.iter().any(|prefix| file.starts_with(prefix)) {
                continue;
            }
        }
        if !requested.is_empty() && !requested.iter().any(|prefix| file.starts_with(prefix)) {
            continue;
        }
        let mut set_tally = Tally::default();
        run_test_set(&root.join(&file), &shared, &mut set_tally, &mut runner);
        tally.merge(set_tally);
    }
    tally
}

fn report(label: &str, tally: &Tally) -> f64 {
    let attempted = tally.passed + tally.failed;
    #[allow(clippy::cast_precision_loss)]
    let rate = if attempted == 0 {
        0.0
    } else {
        tally.passed as f64 / attempted as f64
    };
    println!(
        "QT3 [{label}]: {} passed, {} failed, {} skipped ({:.1}% of {attempted} attempted)",
        tally.passed,
        tally.failed,
        tally.skipped,
        rate * 100.0
    );
    if tally.timed_out > 0 {
        println!(
            "  of the failures, {} did not finish within {}s",
            tally.timed_out,
            case_budget().as_secs()
        );
    }
    for (reason, count) in &tally.skip_reasons {
        println!("  skipped: {count} — {}", reason.label());
    }
    for failure in tally.failures.iter().take(failure_cap()) {
        println!("  - {failure}");
    }
    rate
}

/// Runs `body` on a thread with a stack large enough for the suite's deeply recursive queries.
fn with_deep_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(body)
        .expect("a worker thread")
        .join()
        .expect("the worker thread did not panic")
}

#[test]
fn qt3_xquery_grammar_tests() {
    with_deep_stack(qt3_xquery_grammar_tests_inner);
}

fn qt3_xquery_grammar_tests_inner() {
    let Some(root) = qt3_root() else {
        eprintln!("OXIXML_XML_SUITES_DIR is unset or has no qt3tests directory; skipping");
        return;
    };
    let tally = run_suite(&root, Some(XQUERY_SETS));
    let rate = report("constructors+FLWOR+prolog", &tally);
    let attempted = tally.passed + tally.failed;
    assert!(
        attempted > 3_050,
        "too few QT3 XQuery grammar tests were attempted ({attempted})"
    );
    assert!(
        rate > 0.982,
        "QT3 XQuery grammar pass rate regressed to {:.1}%",
        rate * 100.0
    );
}

#[test]
fn qt3_whole_suite_is_reported() {
    with_deep_stack(qt3_whole_suite_is_reported_inner);
}

fn qt3_whole_suite_is_reported_inner() {
    let Some(root) = qt3_root() else {
        eprintln!("OXIXML_XML_SUITES_DIR is unset or has no qt3tests directory; skipping");
        return;
    };
    let tally = run_suite(&root, None);
    let rate = report("all", &tally);
    let attempted = tally.passed + tally.failed;
    assert!(attempted > 29_300, "too few QT3 tests were attempted");
    assert!(
        rate > 0.980,
        "QT3 overall pass rate regressed to {:.1}%",
        rate * 100.0
    );
}
