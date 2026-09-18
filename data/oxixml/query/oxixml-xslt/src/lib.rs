//! `oxixml-xslt` — an XSLT 1.0 processor with the essentials of XSLT 2.0 and 3.0, in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate occupies the place Apache
//! Xalan and Saxon occupy elsewhere: it compiles a stylesheet into template rules, patterns and
//! instructions, and runs them over the XDM of [`oxixml_infoset`] with [`oxixml_xpath`] as its
//! expression engine and [`oxixml_fn`] as its function library.
//!
//! # Compile once, transform many times
//!
//! ```
//! use oxixml_infoset::Tree;
//! use oxixml_xslt::Stylesheet;
//!
//! let stylesheet = Stylesheet::compile_str(
//!     r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
//!          <xsl:output method="xml" omit-xml-declaration="yes"/>
//!          <xsl:template match="/catalog"><titles count="{count(book)}"><xsl:apply-templates
//!              select="book"><xsl:sort select="@year" data-type="number"/>
//!            </xsl:apply-templates></titles></xsl:template>
//!          <xsl:template match="book"><title year="{@year}"><xsl:value-of
//!            select="title"/></title></xsl:template>
//!        </xsl:stylesheet>"#,
//! )?;
//!
//! let source = Tree::parse_str(
//!     r#"<catalog>
//!          <book year="1965"><title>Dune</title></book>
//!          <book year="1815"><title>Emma</title></book>
//!        </catalog>"#,
//! )?;
//!
//! let result = stylesheet.transform(&source.root())?;
//! assert_eq!(
//!     result.as_str(),
//!     r#"<titles count="2"><title year="1815">Emma</title><title year="1965">Dune</title></titles>"#
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What is here
//!
//! | Concern | Type |
//! |---|---|
//! | Compiling | [`Stylesheet`], [`CompiledStylesheet`], [`StaticOptions`] |
//! | Running | [`CompiledStylesheet::transform`], [`Parameters`], [`Transformation`] |
//! | Output | [`OutputProperties`], [`OutputMethod`], [`serialize`] |
//! | Result tree | [`ResultTree`] |
//! | External resources | [`Resolver`], [`MemoryResolver`], [`FileResolver`] |
//! | Errors | [`XsltError`], [`ErrorPhase`] |
//!
//! # The XSLT this processor implements
//!
//! **XSLT 1.0, complete.** `xsl:stylesheet` and `xsl:transform`, `xsl:import` and
//! `xsl:include` with import precedence, template rules with match patterns, named templates,
//! modes and priorities, `xsl:apply-templates`, `xsl:call-template`, `xsl:apply-imports`,
//! `xsl:param`/`xsl:variable`/`xsl:with-param`, `xsl:if`, `xsl:choose`, `xsl:for-each`,
//! `xsl:value-of`, `xsl:copy`, `xsl:copy-of`, `xsl:element`, `xsl:attribute`, `xsl:text`,
//! `xsl:comment`, `xsl:processing-instruction`, attribute value templates, `xsl:number` in all
//! its levels and formats, `xsl:sort` with several keys, `xsl:key` and `key()`, `xsl:output`,
//! `xsl:decimal-format`, `xsl:strip-space`/`xsl:preserve-space`, `xsl:attribute-set`,
//! `xsl:namespace-alias`, `xsl:message`, `xsl:fallback`, the built-in template rules and the
//! whitespace-stripping rules.
//!
//! **XSLT 2.0 and 3.0 essentials.** `xsl:for-each-group` in all four grouping modes,
//! `xsl:function`, `xsl:analyze-string`, `xsl:sequence` and the `as` attribute,
//! `xsl:result-document`, `xsl:mode` with `on-no-match`, `xsl:next-match`, tunnel parameters,
//! `xsl:iterate` with `xsl:break` and `xsl:next-iteration`, `xsl:perform-sort`,
//! `xsl:on-empty`/`xsl:on-non-empty`/`xsl:where-populated`, `xsl:character-map`,
//! `xsl:try`/`xsl:catch` with the `err:` variables in scope, `xsl:assert`, the
//! `xsl:initial-template` entry point, and the `xhtml` output method. Match patterns cover the
//! full 3.0 grammar: `except`, `intersect`, parenthesized sub-patterns, variable-rooted
//! patterns and patterns written with any axis are matched by the specification's own
//! definition when the step-chain form does not reach them.
//!
//! **Documented gaps.** Streaming (`xsl:stream`, `streamable` modes, `xsl:merge`, `xsl:fork`,
//! `xsl:source-document`, accumulators) is not implemented this cycle; nor is schema-awareness
//! (`xsl:import-schema`, `validation`, type annotations), packaging (`xsl:package`,
//! `xsl:use-package`, `xsl:override`, `xsl:expose`, `xsl:accept`), `xsl:evaluate`, and the
//! `json` and `adaptive` serialization methods with their `build-tree="no"` raw output and
//! `parameter-document`. A stylesheet that reaches one of those is rejected with this crate's
//! own code `OXIXML0001`, so that "not implemented" is distinguishable from "not valid"; an
//! unimplemented instruction with an `xsl:fallback` child runs that instead.
//!
//! # Reaching outside the stylesheet
//!
//! `xsl:include`, `xsl:import`, `document()`, `fn:doc()` and `fn:unparsed-text()` all need
//! something a library cannot invent, so nothing is retrievable until a [`Resolver`] is
//! installed:
//!
//! ```
//! use oxixml_infoset::Tree;
//! use oxixml_xslt::{MemoryResolver, StaticOptions, Stylesheet};
//! use std::sync::Arc;
//!
//! let resolver = Arc::new(MemoryResolver::new().with_document("urn:extra", "<extra>hi</extra>"));
//! let stylesheet = Stylesheet::compile_str_with(
//!     r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
//!          <xsl:output omit-xml-declaration="yes"/>
//!          <xsl:template match="/"><out><xsl:value-of
//!            select="document('urn:extra')/extra"/></out></xsl:template>
//!        </xsl:stylesheet>"#,
//!     StaticOptions::new().with_resolver(resolver),
//! )?;
//! let source = Tree::parse_str("<in/>")?;
//! assert_eq!(stylesheet.transform_to_string(&source.root())?, "<out>hi</out>");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Conformance
//!
//! The crate is tested against the W3C *XSLT 3.0* test suite (`xslt30-test`), whose catalog
//! defines several hundred test sets. The harness runs the ones that do not exercise a
//! documented gap — everything except streaming, packaging, schema-awareness, higher-order
//! functions, `xsl:evaluate`, `xsl:merge`, accumulators and the JSON functions, plus
//! `unicode-90` and `normalize-unicode`, whose cases do not terminate in this processor. The
//! `try` test set is outside the measured allowlist as well, but not as a gap: `xsl:try`,
//! `xsl:catch` and the `err:` variables are implemented and run, and what the set needs before
//! it can be measured honestly is a harness that loads each case's declared `<environment>`. A
//! case
//! inside those sets is *skipped*, not failed, in three documented situations: its
//! `<dependencies>` name a feature listed above as a gap; its stylesheet uses a construct the
//! compiler rejects with `OXIXML0001`; or it is pinned to `XSLT10` or `XSLT20` alone, asserting
//! behaviour a later version deliberately changed — `match="$v"` is a static error in 2.0 and
//! legal in 3.0, and this processor implements 3.0.
//!
//! Exact pass rates move with every conformance fix and are therefore tracked in `CHANGELOG.md`
//! rather than here, where a number would otherwise go stale the next time either the allowlist
//! or the implementation changes; a run against a wider allowlist necessarily reports a lower
//! headline rate than a narrower one that measures only the areas conformance work has already
//! concentrated on, so a bare percentage is meaningless without also stating which allowlist it
//! is over.
//!
//! Set `OXIXML_XML_SUITES_DIR` to a directory holding the suite to run it; the conformance
//! tests skip silently when the variable is unset.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod api;
mod avt;
mod compile;
mod content;
mod error;
pub mod names;
mod output;
mod pattern;
mod pattern30;
mod resolve;
mod run;
mod serialize;
mod tree;
mod typing;
mod validate;

pub use api::{CompiledStylesheet, Parameters, StaticOptions, Stylesheet, Transformation};
pub use compile::{StaticEnv, XsltVersion};
pub use error::{ERROR_NAMESPACE, ErrorPhase, XsltError, XsltResult};
pub use names::XSLT_NAMESPACE;
pub use output::{OutputMethod, OutputProperties, Standalone};
pub use pattern::Pattern;
pub use resolve::{FileResolver, MemoryResolver, Resolver, resolve_uri};
pub use run::DEFAULT_STACK_SIZE;
pub use serialize::serialize;
pub use tree::ResultTree;
