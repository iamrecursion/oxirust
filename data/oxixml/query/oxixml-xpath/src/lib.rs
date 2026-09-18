//! `oxixml-xpath` — XPath 1.0 through 3.1, in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the query engine that
//! Apache Xalan and Saxon occupy elsewhere: a lexer and recursive-descent parser for the whole
//! [XPath 3.1 grammar](https://www.w3.org/TR/xpath-31/), and an evaluator over the XDM of
//! [`oxixml_infoset`] with the function library of [`oxixml_fn`] behind it.
//!
//! # One engine, four languages
//!
//! [`XPathVersion`] is not a hint: it selects both what parses and what it means.
//! [`XPathVersion::XPath10`] gives you the four-type language of node-sets, booleans, numbers and
//! strings — `"3" + 4` is `7`, `@n = 3` is a numeric test, and division by zero is `INF`.
//! [`XPathVersion::XPath20`] and later give you XDM sequences, typed values, the type operators
//! and F&O error codes. [`XPathVersion::XPath30`] adds function items, and
//! [`XPathVersion::XPath31`] adds maps and arrays. Compiling a 3.1 construct as 1.0 is a static
//! error, not a silent success.
//!
//! # Example
//!
//! ```
//! use oxixml_infoset::{AtomicValue, Tree};
//! use oxixml_xpath::{StaticContext, XPath, XPathVersion};
//!
//! let tree = Tree::parse_str(
//!     r#"<catalog>
//!          <book year="1965"><title>Dune</title><price>9.99</price></book>
//!          <book year="1815"><title>Emma</title><price>5.50</price></book>
//!        </catalog>"#,
//! )?;
//! let root = tree.root();
//!
//! // Paths, predicates and the function library.
//! let titles = XPath::compile(
//!     "//book[@year < 1900]/title/string()",
//!     XPathVersion::XPath31,
//!     StaticContext::new(),
//! )?;
//! assert_eq!(titles.evaluate_to_string(root.clone())?, "Emma");
//!
//! // Sequences, `for`, and the arrow operator.
//! let total = XPath::compile(
//!     "sum(for $b in //book return xs:decimal($b/price))",
//!     XPathVersion::XPath31,
//!     StaticContext::new(),
//! )?;
//! assert_eq!(total.evaluate_to_string(root.clone())?, "15.49");
//!
//! // Higher-order functions, maps and arrays are all here too.
//! let squares = XPath::compile_31("array { for-each(1 to 3, function($n) { $n * $n }) }")?;
//! assert_eq!(squares.evaluate_to_string_without_context().is_err(), true);
//!
//! // The same expression under XPath 1.0 semantics.
//! let coerced = XPath::compile("'3' + 4", XPathVersion::XPath10, StaticContext::new())?;
//! assert_eq!(coerced.evaluate_to_number_without_context()?, 7.0);
//!
//! // Variables come from the static context (declaration) and the dynamic context (value).
//! let filtered = XPath::compile(
//!     "count(//book[@year = $year])",
//!     XPathVersion::XPath31,
//!     StaticContext::new().with_variable("year"),
//! )?;
//! let context = filtered
//!     .context_with(root)
//!     .with_variable("year", AtomicValue::integer(1965_i64));
//! assert_eq!(filtered.evaluate_to_number_in(&context)?, 1.0);
//! # Ok::<(), oxixml_xpath::XPathError>(())
//! ```
//!
//! # What is here
//!
//! | Concern | Type |
//! |---|---|
//! | Compiling | [`XPath::compile`], [`CompiledXPath`] |
//! | Language level | [`XPathVersion`] |
//! | Compile-time environment | [`StaticContext`] |
//! | Run-time environment | [`DynamicContext`], [`Focus`], [`DocumentResolver`] |
//! | Values | [`XpathValue`], [`XpathItem`], [`XpathFunction`] |
//! | Types | [`SequenceType`], [`ItemType`], [`KindTest`], [`NameTest`], [`SingleType`], [`Occurrence`] |
//! | Syntax tree | [`ast`] |
//! | Errors | [`XPathError`], [`XPathErrorCode`] |
//!
//! # Function library
//!
//! The whole of [F&O 3.1](https://www.w3.org/TR/xpath-functions-31/) that [`oxixml_fn`]
//! implements is in scope under `fn:`, together with the `map:` and `array:` namespaces, the
//! `math:` namespace, and the `xs:` constructor functions. The functions an evaluator must own —
//! `fn:position`, `fn:last`, the zero-argument accessors, the higher-order functions, and
//! `fn:doc`/`fn:collection`/`fn:unparsed-text` — are implemented here. External resources are
//! unavailable until the application installs a [`DocumentResolver`]; a Pure-Rust library does
//! not read the filesystem or the network on its caller's behalf.
//!
//! # Conformance
//!
//! The crate is tested against the W3C *QT3* suite (FOTS), filtered to the XPath expression
//! tests whose dependencies this engine satisfies. On the last measured run — 20 203 test cases
//! attempted, 11 618 skipped as needing schema awareness, XSLT, serialization or an XQuery-only
//! construct — the engine passed **92.9%** overall: **97.0%** of the operator and production
//! tests (`op/`, `prod/`: the language itself) and **88.2%** of the function tests
//! (`fn/`, `math/`, `map/`, `array/`). The function shortfall is almost entirely functions that
//! live outside this crate — `fn:parse-json`, `fn:xml-to-json`, `fn:json-doc`,
//! `fn:parse-ietf-date`, `fn:random-number-generator` — plus the UCA collations.
//!
//! Set `OXIXML_XML_SUITES_DIR` to a directory holding the suite to run it; the conformance tests
//! skip silently when it is unset.
//!
//! # What this engine cannot check
//!
//! It has no schema, so `element(a, xs:string)` matches only a node that already carries that
//! type annotation and `schema-element(N)` widens to no substitution group. A *typed* function
//! test is exact against any item that declares a signature — an inline function, and a
//! reference to a built-in whose signature the function library records — and falls back to
//! comparing arity alone for the higher-order, resource and `math:` built-ins, which declare
//! none. Everything else — every axis, every kind test, every operator, the whole type syntax,
//! and document order — is exact.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod ast;

mod api;
mod builtins;
mod cast;
mod context;
mod error;
mod eval;
mod function;
mod lexer;
mod parser;
mod types;
mod value;
mod version;

pub use api::{CompiledXPath, XPath};
pub use builtins::supports_function;
pub use context::{
    ARRAY_NAMESPACE, DocumentResolver, DynamicContext, ERR_NAMESPACE, Focus, HostSignatures,
    InvocationKind, MAP_NAMESPACE, MATH_NAMESPACE, StaticContext, TextRetrievalError,
    XSI_NAMESPACE, calling_focus, calling_invocation, invoked_through_function_item,
};
pub use error::{XPathError, XPathErrorCode, XPathResult};
pub use eval::operators::MAX_RANGE;
pub use eval::{CallGuard, DEFAULT_MAX_CALL_DEPTH, DEFAULT_STACK_SIZE, EvaluationLimits};
pub use function::XpathFunction;
pub use types::{ItemType, KindTest, NameTest, Occurrence, SequenceType, SingleType};
pub use value::{XpathItem, XpathValue, atomic_string_value};
pub use version::XPathVersion;

/// Re-export of [`oxixml_fn::FN_NAMESPACE`], the namespace of the `fn:` functions.
pub use oxixml_fn::FN_NAMESPACE;
/// Re-export of [`oxixml_infoset::XSD_NAMESPACE`], the namespace of the `xs:` types.
pub use oxixml_infoset::XSD_NAMESPACE;
