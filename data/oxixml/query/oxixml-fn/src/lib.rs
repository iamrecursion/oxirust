//! `oxixml-fn` — the W3C *XPath and XQuery Functions and Operators 3.1* library, in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the shared function and
//! operator substrate that `oxixml-xpath`, `oxixml-xquery` and `oxixml-xslt` all evaluate on top
//! of: it implements [F&O 3.1](https://www.w3.org/TR/xpath-functions-31/) over the XDM values of
//! [`oxixml_infoset`], and nothing else. There is no parser and no expression tree here — a
//! *function library* is exactly what this is.
//!
//! # The shape of the library
//!
//! Each area of the specification is a module, and each function is a plain Rust function with
//! the types its signature implies. `fn:substring` takes a `&str` and two `f64`s, `fn:max` takes
//! a [`Sequence`](oxixml_infoset::Sequence) and a collation, `fn:format-number` takes an
//! [`AtomicValue`](oxixml_infoset::AtomicValue) and a picture string. That keeps the library
//! usable directly, without going through a dynamic argument list:
//!
//! | Module | Specification | Contents |
//! |---|---|---|
//! | [`accessors`] | §5 | `fn:node-name`, `fn:nilled`, `fn:string`, `fn:data`, `fn:base-uri`, `fn:document-uri` |
//! | [`numeric`] | §4, §8.4 | the arithmetic operators and promotion, `fn:abs` … `fn:round-half-to-even`, `fn:format-number` |
//! | [`strings`] | §5, §6 | construction, comparison under a collation, the regular-expression functions, the URI functions |
//! | [`boolean`] | §7 | `fn:true`, `fn:false`, `fn:boolean`, `fn:not` |
//! | [`datetime`] | §9 | component extraction, `fn:adjust-*-to-timezone`, `fn:format-dateTime`, `fn:format-integer` |
//! | [`qnames`] | §11 | `fn:resolve-QName`, `fn:QName` and the decomposition functions |
//! | [`nodes`] | §14 | `fn:name`, `fn:lang`, `fn:root`, `fn:id`, `fn:generate-id` |
//! | [`sequences`] | §14 | the general sequence functions and the aggregates |
//! | [`higher_order`] | §16 | `fn:for-each`, `fn:filter`, `fn:fold-left`, `fn:sort`, `fn:apply` |
//! | [`maps`], [`arrays`] | §17 | the whole `map:` and `array:` namespaces |
//! | [`xmlio`] | §14.7 | `fn:parse-xml`, `fn:parse-xml-fragment`, `fn:serialize` |
//! | [`diagnostics`] | §3 | `fn:error` and `fn:trace` |
//! | [`registry`] | — | the name-and-arity catalogue `fn:function-lookup` reads |
//!
//! # Context, collations and errors
//!
//! [`FnContext`] carries the handful of context components F&O §2 says the functions depend on:
//! the default collation and the set of known collations, the static base URI, the implicit
//! timezone, the *stable* current dateTime, the decimal formats, and the destination of
//! `fn:trace`.
//!
//! [`Collation`] is the pluggable string comparison every collation-sensitive function consults.
//! Four are built in — the mandatory [`CODEPOINT_COLLATION`], the W3C
//! [`HTML_ASCII_CASE_INSENSITIVE_COLLATION`], and a Unicode case-folding collation under both
//! [`CASELESS_COLLATION`] and [`CASEBLIND_COLLATION`] — and any other can be registered. The
//! `http://www.w3.org/2013/collation/UCA` URIs of XSLT 3.0 §5.3.4 are resolved on demand by
//! [`UcaCollation`], which implements the level structure of UTS #10 over a root ordering of its
//! own, with the `lang`, `strength`, `alternate`, `maxVariable`, `backwards`, `caseLevel`,
//! `caseFirst`, `numeric` and `fallback` parameters honoured and tailorings for German and
//! Swedish; the repertoire that carries *explicit* weights, and the implicit weights everything
//! else takes, are documented on the [`uca`] module.
//!
//! Every failure is a [`FnError`] carrying the qualified error name a conforming processor would
//! raise, so `err:FORG0001`, `err:FOCH0002` and `err:FODF1310` mean here exactly what they mean
//! in an XQuery `try`/`catch`.
//!
//! # Example
//!
//! ```
//! use oxixml_fn::{datetime, nodes, numeric, sequences, strings, xmlio, FnContext, FnErrorCode};
//! use oxixml_infoset::{AtomicType, AtomicValue, Sequence, XdmNode};
//!
//! let context = FnContext::new();
//!
//! // Strings, with the collation-sensitive functions honouring the context.
//! assert_eq!(strings::substring("metadata", 4.0, Some(3.0)), "ada");
//! assert_eq!(strings::upper_case("abCd0"), "ABCD0");
//! assert!(strings::contains(&context, "tattoo", "tat", None)?);
//! assert_eq!(strings::tokenize("1, 15, 24", ",\\s*", "")?, ["1", "15", "24"]);
//!
//! // Numbers, with XPath's promotion rules and its picture strings.
//! let sum = numeric::add(
//!     &AtomicValue::integer(1_i64),
//!     &AtomicValue::parse(AtomicType::Decimal, "2.5")?,
//! )?;
//! assert_eq!(sum.string_value(), "3.5");
//! assert_eq!(
//!     numeric::format_number(
//!         Some(&AtomicValue::parse(AtomicType::Decimal, "12345.6")?),
//!         "#,##0.00",
//!         context.default_decimal_format(),
//!     )?,
//!     "12,345.60"
//! );
//!
//! // Dates, with the F&O picture-string language.
//! let instant = AtomicValue::parse(AtomicType::DateTime, "2002-12-31T19:20:00-05:00")?;
//! assert_eq!(datetime::year_from_date_time(&instant)?, 2002);
//! assert_eq!(
//!     datetime::format_date_time(&context, &instant, "[FNn], [D1o] [MNn] [Y]", None, None, None)?,
//!     "Tuesday, 31st December 2002"
//! );
//!
//! // Nodes and sequences, over a parsed document.
//! let document = xmlio::parse_xml("<books><book>Dune</book><book>Emma</book></books>", None)?;
//! let root = document.children().into_iter().next().expect("a document element");
//! assert_eq!(nodes::local_name(&root), "books");
//! let titles = Sequence::from_iter(root.children());
//! assert_eq!(sequences::count(&titles), 2);
//! assert_eq!(strings::string_join(&titles, ", ")?, "Dune, Emma");
//!
//! // ... and errors carry the specification's codes.
//! let failure = sequences::exactly_one(&titles).expect_err("two items");
//! assert!(failure.is(FnErrorCode::Forg0005));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Scope
//!
//! Deliberately outside this crate: the external-resource functions (`fn:doc`, `fn:collection`,
//! `fn:unparsed-text`, `fn:json-doc`), which need an I/O policy an evaluator owns; and `math:*`,
//! which is a self-contained numerics module.
//!
//! The JSON functions of §17.5 and §17.6 — `fn:parse-json`, `fn:json-to-xml` and
//! `fn:xml-to-json` — *are* here, in [`json`], because they are pure functions over strings and
//! XDM values with no I/O of their own. Their `fallback` option takes a function item, which the
//! data model cannot hold, so it is passed to [`json::JsonOptions`] as a Rust closure that an
//! evaluator supplies.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod accessors;
pub mod arrays;
pub mod boolean;
pub mod collation;
pub mod context;
pub mod datetime;
pub mod deep_equal;
pub mod diagnostics;
pub mod error;
pub mod function;
pub mod higher_order;
pub mod json;
pub mod maps;
pub mod nodes;
pub mod numeric;
pub mod qnames;
pub mod registry;
pub mod sequences;
pub mod signature;
pub mod strings;
pub mod uca;
pub mod xmlio;

pub use collation::{
    CASEBLIND_COLLATION, CASELESS_COLLATION, CODEPOINT_COLLATION, CaseblindCollation,
    CaselessCollation, CodepointCollation, Collation, Collations,
    HTML_ASCII_CASE_INSENSITIVE_COLLATION, HtmlAsciiCaseInsensitiveCollation,
};
pub use context::{CollectingTrace, FnContext, StderrTrace, TraceSink};
pub use deep_equal::{deep_equal_with, items_equal};
pub use error::{ERROR_NAMESPACE, FnError, FnErrorCode, FnResult};
pub use function::FunctionItem;
pub use registry::{FN_NAMESPACE, FunctionRegistry, builtins};
pub use uca::{UCA_COLLATION_BASE, UcaCollation};
