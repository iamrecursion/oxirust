# Changelog

All notable changes to the OxiXML workspace are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) · versioning per COOLJAPAN branch policy.

## 0.1.3 - Unreleased

## 0.1.2 - 2026-08-10

### Breaking

**`oxrdf` / `oxttl` API parity**

`oxixml-model`'s documentation has always claimed that code written against `oxrdf` ports over by
changing only the `use` path. It was not true: the crate handed out owned terms where `oxrdf`
hands out borrows, which broke on the first `for triple in &graph`. These changes close that gap,
checked against `oxrdf` 0.3.3, `oxttl` 0.2.3, `oxrdfxml` 0.2.3, `oxjsonld` 0.2.5, `oxrdfio` 0.2.5
and `spargebra` 0.4.6. Every signature below now matches its upstream counterpart.

- `oxixml-model` — **iteration yields borrows.** `Graph::iter`, `IntoIterator for &Graph`,
  `Dataset::iter`, `IntoIterator for &Dataset`, and every `GraphView` / `GraphViewMut` iterator
  now yield `TripleRef<'_>` / `QuadRef<'_>` instead of owned `Triple` / `Quad`. The pattern
  queries follow: `triples_for_subject`/`_predicate`/`_object` and `triples_for_pattern` yield
  `TripleRef<'_>`, `objects_for_subject_predicate` yields `TermRef<'_>`,
  `predicates_for_subject_object` yields `NamedNodeRef<'_>`, `subjects_for_predicate_object`
  yields `NamedOrBlankNodeRef<'_>`, and the singular `object_for_subject_predicate` /
  `subject_for_predicate_object` return the matching `Option<…Ref<'_>>`. Call `into_owned()` where
  an owned term is needed. The interning layer now decodes by borrowing from its string table, so
  iteration allocates nothing at all.
- `oxixml-model` — **mutators take borrows.** `Graph::insert`, `Dataset::insert` and
  `GraphViewMut::insert` take `impl Into<TripleRef<'a>>` / `impl Into<QuadRef<'a>>`, matching the
  `remove` and `contains` they sit next to; an owned value is now passed as `&triple`.
  `Dataset::graph_mut` takes `impl Into<GraphNameRef<'_>>` rather than `impl Into<GraphName>`, and
  the blanket `FromIterator`/`Extend` impls are split into an owned pair and a borrowed pair, as
  upstream has them.
- `oxixml-model` — **vocabulary constants are borrowed.** `vocab::{rdf, rdfs, xsd, geosparql}`
  constants are `NamedNodeRef<'static>` (`Copy`) instead of owned `NamedNode`, which is what lets
  `rdf::TYPE` be passed by value into any `impl Into<NamedNodeRef<'_>>` slot. Code that needs an
  owned node writes `rdf::TYPE.into_owned()`. (`oxixml-vocab` already used this shape.)
  Consequently `Literal::datatype` returns `NamedNodeRef<'_>` rather than `&NamedNode`.
- `oxixml-model` — **`&str` constructors accept any lifetime.** `NamedNode::new`,
  `NamedNode::new_unchecked`, `BlankNode::new`, `Variable::new`, `Literal::new_simple_literal`,
  `Literal::new_typed_literal` and the language-tagged constructors are bound by the new
  `IntoOxString` trait instead of `Into<OxString>`. `OxString` is `OxStr<'static>`, so the old
  bound rejected any `&str` that did not live for `'static` — the single most common thing a port
  from `oxrdf`'s `impl Into<String>` would hit. Borrowed input is copied; `OxString` and `String`
  input is moved without a new allocation, so the existing zero-copy paths are unchanged.
- `oxixml-model` — `Dataset::canonicalize_blank_nodes` and its `try_` variants return
  `HashMap<BlankNodeRef<'_>, BlankNode>` instead of `HashMap<BlankNode, BlankNode>`.
- `oxixml-turtle`, `oxixml-rdfxml`, `oxixml-jsonld`, `oxixml-trix`, `oxixml-io` — **serializers
  take borrows.** Every `serialize_triple` / `serialize_quad` (blocking, async and low-level)
  takes `impl Into<TripleRef<'a>>` / `impl Into<QuadRef<'a>>` instead of `&Triple` / `&Quad`.
  Existing `&owned` call sites keep compiling unchanged. `oxixml-io`'s internal default-graph
  extraction now returns a `TripleRef` and no longer clones the terms it forwards.

Three deviations from `oxrdf` remain, now stated in `oxixml-model`'s crate documentation rather
than glossed over. None changes what a term means. `Term` additionally implements `PartialEq`
against `NamedNode`/`BlankNode`/`Literal`, which makes `term == x.into()` ambiguous and requires
naming the target as `term == Term::from(x)`; the owned accessors return `OxString` rather than
`String`; and `oxrdf`'s `#[deprecated]` `is_plain` / `LiteralRef::destruct` are not provided.

**Builder options take `impl Into<String>`**

`with_prefix` and `with_base_iri` took `&str`, where every upstream counterpart takes
`impl Into<String>`: a `String` built for the occasion could not be moved in, and generic code
holding one had to borrow it back down. Both now take `impl Into<String>` on `TurtleParser` /
`TurtleSerializer`, `TriGParser` / `TriGSerializer`, `N3Parser`, `RdfXmlParser` /
`RdfXmlSerializer`, `JsonLdParser`, and `oxixml-io`'s `RdfParser` / `RdfSerializer`, checked
against `oxttl` 0.2.3, `oxrdfxml` 0.2.3, `oxrdfio` 0.2.5 and `oxjsonld` 0.2.5.
`JsonLdSerializer` already had the upstream signature. N-Triples and N-Quads have no notion of a
prefix in either processor, so neither gains a `with_prefix`.

- `oxixml-io`: `RdfParser::with_document_loader` now requires its closure to be
  `Send + Sync + UnwindSafe + RefUnwindSafe + 'static`, the bound `oxrdfio` states. It required
  `'static` alone, which is what let a non-`Send` loader reach the JSON-LD parsers.

**`oxixml-jsonld` parsers are `Send`**

`ReaderJsonLdParser`, `SliceJsonLdParser` and `TokioAsyncReaderJsonLdParser` held their document
loader as `Box<dyn DocumentLoader>`, so none of them was `Send` and none could be moved onto a
worker thread — the ordinary way to read a document off the request path. The box is now
`Box<dyn DocumentLoader + Send>`, and `with_loader` / `with_load_document_callback` require `Send`
to match. `oxjsonld` puts the same bound on its own `LoadDocumentCallback`; the `DocumentLoader`
trait itself is unchanged, so a hand-written loader needs no edit unless it is genuinely
thread-hostile.

### Fixed

- `oxixml-jsonld`: `JsonLdSerializer::with_prefix` and `with_base_iri` were validated, stored, and
  then ignored — no `@context` was ever written, and the base never reached the output. They now
  produce the document `oxjsonld` produces: `{"@context": {…}, "@graph": […]}`, with `@base` first
  and the prefixes in name order (the empty prefix name written as `@vocab`), and node identifiers
  made relative to the base. Property keys stay expanded IRIs, exactly as upstream leaves them —
  declaring a prefix declares a term, it does not shorten the keys. `@type` is never made relative,
  because expansion resolves a type against `@vocab` rather than `@base`. With neither option set
  the output is unchanged. The one place the two processors genuinely differ — this crate runs the
  specified *fromRdf* algorithm, which folds `rdf:type` into the `@type` keyword, while `oxjsonld`
  streams it out as an ordinary property key — is now stated in the crate documentation instead of
  being left for a reader to discover.
  `oxixml-io`'s `RdfSerializer` forwards `with_prefix` and `with_base_iri` to the JSON-LD
  serializer, where `oxrdfio` drops both, so a JSON-LD document written through the umbrella crate
  carries the `@context` envelope where `oxrdfio` writes a bare array. Dropping a caller's option
  is the defect being fixed here, so the forwarding stays; both methods now say so.
- `oxixml-turtle`: `N3Term` gained the `From<NamedNodeRef<'_>>` impl `oxttl` has, so a vocabulary
  constant can be converted directly. Its `NamedNodeRef` was imported only under the `rdf-12`
  feature while the impl is unconditional, which stopped the crate compiling in its default
  configuration; the import is now unconditional too.

## 0.1.1 - 2026-08-06

A hardening release. It finishes the query/transform tier against its W3C suites, adds a typed
SVG document model, and makes arbitrary-precision `xs:integer`/`xs:decimal` the default. It
breaks compatibility with 0.1.0 deliberately and ships no compatibility shims: 0.1.0 is
superseded and is to be yanked from crates.io when 0.1.1 goes out, so 0.1.1 is the version to
depend on.

### Breaking

Two of these change a public signature. The rest change what a correct call *answers*, which in
practice matters more — each names the rule it now follows.

**Public API**

- `oxixml-hdt`: `BitmapTriples::write` now returns `Result` (was infallible), and a new
  `HdtError::NonDenseSubjectIds` variant is returned rather than silently renumbering a
  gapped-subject structure; the SPO subject axis now handles non-dense subject ids correctly.
  Re-exported via the `oxixml` facade. No in-workspace caller is affected — all go through
  `HdtWriter::write`, which was already fallible.
- `oxixml-dsig` `KeyInfoItem`: the `DsaKeyValue` and `EcKeyValue` variants changed from unit
  variants to struct variants carrying parsed key material (`DsaKeyValue { p, q, g, y }`,
  `EcKeyValue { named_curve, public_key }`); code that pattern-matches on those two must be
  updated. (`RsaKeyValue` was already a struct variant at 0.1.0 and is unchanged.)

**Behaviour**

- `oxixml-regex`: `Regex::xpath` reads character classes against **XSD 1.0** — the version this
  processor implements and reports, since `system-property('xsl:xsd-version')` answers `1.0`.
  F&O 3.1 §5.6.1 defines the XPath regular-expression syntax by reference to the XML Schema
  grammar, so which appendix applies is a property of the processor rather than of the call. A
  host that implements XSD 1.1 compiles with the new
  `Regex::xpath_versioned(pattern, flags, version)` instead. `Regex::xsd` is unchanged: a
  `pattern` facet is read against the version of the schema document that carries it, which the
  validator already passes explicitly to `Regex::xsd_versioned`.
- `oxixml-fn`: `fn:xml-to-json` raises `FOJS0006` for a document node with two element children
  instead of silently taking the first one. A parsed document can never have that shape; an
  XQuery fragment and an XSLT temporary tree both can.
- `oxixml-fn` / `oxixml-xpath`: list-valued lexical forms separate on **XML white space**
  (`#x20 #x9 #xD #xA`), not on Unicode white space. This governs `cast as`, `castable as` and
  the `xs:IDREFS` / `xs:NMTOKENS` / `xs:ENTITIES` constructors (XPath 3.1 §19.3.4), the
  `cdata-section-elements` and `suppress-indentation` serialization parameters, and the
  pseudo-attributes of a text declaration read by `fn:parse-xml-fragment`. A value separated by
  a non-XML Unicode space such as U+00A0 used to be split into several tokens and now stays one,
  which is in no item type's lexical space — so it raises `FORG0001` (or the corresponding
  parameter/parse error) rather than silently producing extra items.
- `oxixml-fn`: `fn:idref` no longer tokenizes its argument. Each supplied string is one
  candidate IDREF value; a candidate holding white space now matches nothing, where before it
  matched as though it had been several candidates.
- `oxixml-xpath`: `KindTest::is_subtype_of` implements the XPath 3.1 §2.5.6.2 subtype-itemtype
  judgement rather than accepting only `node()` and identity, so `instance of`, `treat as` and
  function coercion over function items with a declared signature answer more relations true.
  Linked to it, §2.5.5.3's nilled-element exclusion now applies only to an element test that
  *names a type*: on a schema-validated tree, `element()`, `element(*)` and `element(N)` match a
  nilled element where they did not before.
- `oxixml-infoset` map keys: `same_key` is EXACT value comparison, not double promotion — two
  integers that round to the same `xs:double` stay DISTINCT map keys (QT3 `same-key-026`).
  `eq`/`compare` and `fn:distinct-values` keep promotion semantics, so this bites only for numeric
  keys of different types that used to collapse onto one `xs:double`.
- `oxixml-xslt`: `Transformation::as_sequence()` answers the raw result of a principal
  destination whose output method builds no tree (`json`, `adaptive` — `build-tree="no"`),
  instead of the empty document node such a destination used to stand for. Separately, the
  built-in shallow-copy template rule applied to a **document node** at the top level of a
  sequence now builds a document node of its own, which is what `xsl:copy` already did under the
  same condition (XSLT 3.0 §6.7.3).

### Added

- New crate `trees/oxixml-svg` — a typed SVG 1.1/2.0 document model over `oxixml-dom`
  (`SvgDocument`/`SvgElement`), reference resolution with cycle detection, and a CSS cascade
  with a documented selector subset; a complete microsyntax layer (path data with spec
  error-recovery, transform lists, lengths/angles, colors incl. 149 named, paint,
  viewBox/preserveAspectRatio, points, SMIL clock/timing, style declarations); vocabulary
  catalogs (88 elements, 216 attributes, content models); a geometry engine (bezier/arc math
  per SVG Appendix F, exact bounding boxes, path length/point-at-length, viewBox transforms);
  the `SvgBuilder` programmatic writer with correct escaping + `PathBuilder`; an SVG-aware
  linter (18 diagnostic codes); an `oxixml-cli` `svg` subcommand; and a facade `svg` feature.
- A W3C SVG 1.1 test-suite roundtrip harness (env-discovered, vendored at the shared suites dir).
- Arbitrary-precision `xs:integer` / `xs:decimal`: unbounded integer/decimal arithmetic, big
  numeric literals, casts in both directions, large `to` ranges, and exact map-key / distinct-values
  / index-of / sort semantics; `fn:unsignedLong` now reaches `u64::MAX`. Fixed-width in-range values
  are byte-identical (no format/perf change). New `oxixml-infoset` `BigInteger`/`BigDecimal` atomics.
- XSD 1.1: circular attribute groups now accepted (W3C bug 15795) and four §3.9.6 particle-restriction
  relaxations (emptiable-base occurrence, unbounded-base recurse reuse, single-branch map-and-sum).
- `oxixml-xml`: the serde `Deserializer` now surfaces text-only elements as scalar strings under
  `deserialize_any`, enabling internally-/adjacently-/untagged and flattened enum/struct
  deserialization (8 previously-blocked cases); 7 remain limited by serde#1183 (numeric/bool variant
  recovery from untyped text), documented in-crate.
- `oxixml-dsig` signature algorithms: DSA (`#dsa-sha1`, `xmldsig11#dsa-sha256`) and ECDSA over
  P-256 and P-384 (`xmldsig-more#ecdsa-sha1` through `-sha512`) sign+verify with RFC 6979
  deterministic nonces, and RSASSA-PSS (`xmldsig-more#rsa-pss`) sign+verify.
- `oxixml-dsig` X.509: RFC 5280 §6.1 certification-path validation (signatures, validity dates,
  name chaining, basic constraints, key usage, critical extensions) via a public `PathValidator`.
- `oxixml-enc` key transport/agreement: RSA key generation (`RsaPrivateKey::generate`), and
  ECDH-ES (P-256/P-384) plus finite-field Diffie-Hellman (`dh-es`) key agreement deriving their
  key-encryption key via the ConcatKDF (the legacy `dh` agreement keeps XMLEnc 1.0 §5.5.2's own
  fixed key-derivation function).
- **A cooperative deadline for query and transformation.** `EvaluationLimits::declare_deadline`
  (`oxixml-xpath`) and `Parameters::with_deadline` (`oxixml-xslt`) stop a runaway expression or
  stylesheet with `XPDY0130` — the code the specification reserves for an implementation-defined
  limit — with a message saying the limit is a configurable deadline. It is checked *between*
  steps, at the checkpoints the call-depth budget already uses plus the evaluator's per-item
  loops, so an evaluation with no deadline never reads the clock. It is cooperative rather than a
  hard kill: one builtin that is itself slow runs to completion, and a caller that must reclaim
  the thread regardless still needs its own wall-clock guard.
- `fn:transform` (F&O 3.1 §16.4) in `oxixml-xslt`, which is the crate that can invoke an XSLT
  processor: the options map names a stylesheet, it is compiled with the resolver and base URI
  the calling transformation is using, and the prescribed result map is returned — the principal
  result under `output`, each secondary result under the absolute URI `xsl:result-document` wrote
  it to. Option entries this processor cannot honour are rejected rather than ignored, so a
  stylesheet is never silently given a transformation it did not ask for.
- A declared base output URI — `Parameters::with_base_output_uri` — which `fn:current-output-uri`
  and a relative `xsl:result-document href` are read against; and XSLT 3.0 §24.3's rule that the
  output URI is *absent* while evaluating stylesheet functions, dynamic function calls, global
  variables, stylesheet parameters and patterns.
- The `json` and `adaptive` output methods for `xsl:output` / `xsl:result-document`, including
  `build-tree="no"` destinations whose result is a raw sequence rather than a tree.
- `xsl:apply-imports` and `xsl:next-match` search an import-precedence *window* — at or above the
  floor of the current rule's import subtree and strictly below the current rule — so an
  overridden rule cannot re-match itself.
- `xsl:number` holds an unbounded place-marker value. XSLT 3.0 §12.3 converts `value=` to an
  `xs:integer` and XPath integers are unbounded, so `value="1e100"` now formats its hundred-and-one
  digits instead of saturating an `i64`; a numbering sequence with a range of its own (Roman
  numerals, words, the alphabetic sequences) falls back to decimal notation, which §12.3 licenses
  in as many words.
- DTD-declared `ID`/`IDREF` typing survives whitespace stripping: a tree `xsl:strip-space`
  re-materializes is the same document under XSLT 3.0 §4.4, so `fn:id`, `fn:idref` and `id(…)`
  patterns still resolve DTD-declared identifiers over it (`id-028` … `id-031`).
- Declared `xsl:function` signatures reach the XPath static context
  (`StaticContext::with_host_signatures`), so a function item obtained for a stylesheet function
  carries its declared type and takes part in function coercion and `instance of`.
- Source positions travel with a failure: `oxixml_infoset::ParseOptions::record_source_positions`
  is on for stylesheet compilation, and `XsltError` carries the position of the innermost
  instruction the error passed out of — which is what makes an `xsl:try`/`xsl:catch` diagnostic
  point at a line of the stylesheet.
- `FnContext::with_xml_version` — the XML version in force is part of the function context, so
  serialization and character-class decisions that differ between XML 1.0 and 1.1 follow the
  document rather than a default.
- `oxixml-testsuite`: catalogue `<environment>` reuse and `fn:collection` support across the QT3
  drivers, XSLT 3.0 §3.11 embedded stylesheets (`<source defines-stylesheet="true">`, resolved
  through the `xml-stylesheet` processing instruction and a declared `ID` attribute), and
  XInclude processing for sources that declare it.

### Changed / Fixed

- `oxixml-sparql-syntax` test harness: the manifest scanner now respects Turtle quoted-string / IRIREF
  boundaries; the crate now measures the full corpus (previously silently dropped ~20 entries whose
  descriptions contained a bare `a`). No parser behavior changed.
- Restore XSLT 3.0 §6.6.1 mode-declaration duplicate check to deferred (winning-precedence) semantics
  (a lower-precedence conflict resolved by a later higher-precedence declaration is not an error).
- A round of smaller F&O / engine completions: full Unicode `SpecialCasing` case mapping for
  `fn:upper-case`/`fn:lower-case` (`ß`→`SS`, ligature expansion, Greek final sigma),
  `fn:available-system-properties#0`, `xs:numeric` correctly subtyping `xs:anyAtomicType`,
  effective `xsl:function cache="yes"` memoization, two-phase XML 1.0 §4.4/4.5 `EntityValue`
  semantics unified across `oxixml-dom`/`oxixml-sax`/`oxixml-infoset`, `fn:serialize`'s `json`
  output method now honoring the caller's `max_depth`, and a `CyrillicSupplementary` regex
  block-alias fix.
- **Global variables cost nothing per expression.** The XSLT run-time rebound every global
  variable, by value, into a fresh binding chain for every expression evaluated and every pattern
  matched — so a stylesheet that read a document into a global copied every item of it per
  expression. The globals now live on one shared, immutable binding chain built once and
  retargeted per site. Measured on one binary with the change gated behind an environment
  variable: 2000 stylesheet-function calls over a large global, 6.337 s → 40.7 ms.
- XSLT base-URI rules on copies: a parentless copy keeps the base URI it was copied with, an
  attached copy inherits its new parent's (XSLT 3.0 §11.9.1), and a copied `xml:base` is resolved
  against the *copying* instruction rather than against where the original stood (§11.9.2).
- `fn:snapshot` of an attribute or namespace node that has a parent now succeeds for XPath and
  XQuery callers too, not only through `oxixml-xslt`'s own path: it returns the copy of that node
  on the copy of its parent element. Only a *parentless* attribute, namespace or text node is
  still rejected, which is `fn:copy-of`'s case and is unchanged.
- `oxixml-schema`/`oxixml-testsuite`: the XML Schema Test Suite writes its `version` selector at
  three levels — `testSet`, `testGroup` and the individual test — and the innermost one in force
  governs. The workspace harness read it only on the node it was looking at, so the XSD 1.0 pass
  was assembling `xs:alternative`, `targetNamespace` on `xs:element`, `xs:anyAtomicType` and
  `+INF` under XSD 1.0 and recording the refusals the suite asked for as failures. Measurement
  defect only; no validator behaviour changed, and the two independent runners now agree slot for
  slot.
- `oxixml-testsuite`: a retrieved document's external entities resolve against that document's own
  directory (XML 1.0 §4.2.2), not against the test set's; and a stylesheet module's own external
  general entities are retrieved while the module is parsed.

### Security

- `oxixml-dsig` and `oxixml-enc` both cap RSA public exponents at a new
  `MAXIMUM_RSA_PUBLIC_EXPONENT_BITS` (256 bits), rejecting an implausibly large exponent before
  any `modpow` runs: `modpow`'s cost is linear in the exponent's bit width, so an unbounded
  exponent let a crafted key turn a single verify or key-transport operation into seconds of CPU
  time.

### Conformance

Measured on the 0.1.1 tree, from one binary built once and driven repeatedly, with every
environment variable that changes what is measured left unset. The XSLT catalogue was run four
times — three from one binary and a fourth from a fresh build — and all four agree on every one of
the 14601 dispositions, id for id.

```text
XSLT 3.0    xslt30-test        7340 / 7420 adjudicated passed,  80 failed,  7181 skipped
                               98.9% of adjudicated cases, over the whole 14601-case catalogue
XPath 3.1   QT3/FOTS          21731 / 21731 passed,   0 failed, 10090 skipped   100.0%
XQuery 3.1  QT3/FOTS          29744 / 29744 passed,   0 failed,  2077 skipped   100.0%
XSD 1.0     xsdtests          39271 / 39309 passed,  38 failed,     11 skipped   99.9%
XSD 1.1     xsdtests          41434 / 41463 passed,  29 failed,     11 skipped   99.9%
```

Every failing case of those suites is named, classed and evidenced in an **expected-failure
ledger inside the runner**, and each run asserts that its failing set is exactly that table — in
both directions, so a case that starts failing is a regression and a ledgered case that stops
failing is a stale claim. Both turn the run red. The 147 entries are 75 suite defects (the case
or its expectation is wrong or unsatisfiable, with a citation), 6 declared dependencies
(streaming, or a retrieval over the network), 30 expectations pinned to a Unicode version this
workspace has moved past, and **36 named gaps in this workspace**, listed as roadmap work in
`TODO.md`. The QT3 drivers are gated on zero wrong answers rather than zero failures, since a
per-case timeout measures the machine and a wrong answer measures the processor; neither run
produced one.

The 7181 XSLT skips are labelled individually, because "skipped" hides the difference between a
case written for a processor this one is not and a case this processor cannot yet reach:

```text
4898  the catalogue's own <dependencies> declarations — streaming 2899, XSLT 2.0 only 1097,
      schema-aware 672, backwards compatibility 121, dynamic evaluation 59, and 50 across nine
      smaller labels
1460  cost:unicode-90 — a set the driver declines on a whole-catalogue run and runs on request
      (OXIXML_XSLT30_SETS=unicode-90); nothing puts these cases out of audience
 823  this workspace's own limits, of which 703 are stylesheets the compiler declined
      (704 GAP lines — see the note below), by construct:
         432 xsl:package   53 xsl:accumulator   10 xsl:evaluate
          90 xsl:merge     30 xsl:use-package    7 the parameter-document parameter
          59 xsl:source-document                23 xsl:fork
      and the remaining 120 are 65 transformations that raised before producing a tree,
      35 assert-message cases the xsl:message log is not complete enough to decide,
      13 expectations this parser does not accept as a fragment, and 7 across four smaller labels
```

The construct tally counts the 704 `GAP` lines the trace carries, one per stylesheet the compiler
declined; 703 of them are skipped under `not-implemented:compile`, and the extra one
(`result-document-1406`) reaches its gap at run time and is skipped under
`not-implemented:transform`.

The other suites — RDF, JSON-LD, SPARQL syntax, RELAX NG, Schematron, HTML5, RDFa, Canonical XML,
XMLDSig and xmlconf — are itemised in the 0.1.0 entry below and were not re-measured for this
release.

## 0.1.0 - 2026-07-29

### Added
- Phase 1 foundation: `oxixml-support` (SWAR byte scan, SHA-1/2 + HMAC, BigUint modpow, PRNG),
  `oxixml-unicode` (Unicode 17.0.0 tables, NFC/NFD/NFKC/NFKD, XML name classes),
  `oxixml-encoding` (UTF-8/16, US-ASCII, the WHATWG single-byte set — ISO-8859-2..16,
  windows-125x, KOI8-R/U, IBM866, macintosh, x-mac-cyrillic — plus Shift_JIS, EUC-JP and
  ISO-2022-JP with incremental state machines; XML Appendix-F sniffing),
  `oxixml-text` (base64/hex/percent, ECMAScript + XSD float formatting),
  `oxixml-regex` (XSD + XPath regex flavors), `oxixml-iri` (RFC 3987, oxiri-shaped),
  `oxixml-langtag` (BCP 47, oxilangtag-shaped), `oxixml-xsd` (XSD 1.1 datatypes,
  oxsdatatypes-shaped), `oxixml-qname` (names, namespaces).
- Phase 2 XML core: `oxixml-xml` — a complete quick-xml 0.41-shaped pull parser and writer
  (`Reader`, `NsReader`, `Writer`/`ElementWriter`, events, escaping, `Config`), the serde
  deserializer and serializer behind `serialize`, and the `async-tokio` feature; plus
  `oxixml-quickxml-compat`, a path-compatible drop-in shim for downstream `quick_xml` users.
- Phase 3 RDF base: `oxixml-model` (oxrdf-shaped RDF 1.2 model), `oxixml-json`
  (json-event-parser-shaped streaming JSON), `oxixml-vocab`, `oxixml-turtle`
  (N-Triples/N-Quads/Turtle/TriG/N3), `oxixml-rdfxml`, `oxixml-trix`, `oxixml-jsonld`
  (JSON-LD 1.1 expand/compact/flatten/toRdf/fromRdf), `oxixml-canon` (RDFC-1.0),
  `oxixml-hdt` (HDT binary read/write), `oxixml-io` (unified RDF I/O facade),
  `oxixml-sparql-syntax` (spargebra-shaped SPARQL 1.0/1.1/1.2 query+update to algebra),
  `oxixml-sparql-results` (XML/JSON/CSV/TSV results I/O).
- Phase 4 trees & validation: `oxixml-dom` (DOM Level 3 Core-class arena tree with namespace
  fixup serialization), `oxixml-sax` (SAX2-style push API), `oxixml-infoset` (XDM data model),
  `oxixml-html` (HTML5 tokenizer + tree construction), `oxixml-dtd` (validating DTD processor
  with entity-expansion limits), `oxixml-catalog` (OASIS XML Catalogs 1.1),
  `oxixml-xinclude` (XInclude 1.0/1.1), `oxixml-schema` (XSD 1.0/1.1 validation),
  `oxixml-relaxng` (RELAX NG XML + compact syntax, derivative validation),
  `oxixml-schematron` (ISO Schematron).
- Phase 5 query & transform: `oxixml-fn` (XPath/XQuery F&O 3.1 function library),
  `oxixml-xpath` (XPath 1.0–3.1), `oxixml-xquery` (XQuery 3.1), `oxixml-xslt`
  (XSLT 1.0 complete + 2.0/3.0 essentials).
- Phase 6 security & format: `oxixml-c14n` (Canonical XML 1.0/1.1/Exclusive + 2.0),
  `oxixml-dsig` (XMLDSig sign+verify incl. RSA PKCS#1 v1.5 + HMAC), `oxixml-enc`
  (XMLEnc AES-CBC/GCM + AES-KW + RSA-OAEP/PKCS#1v1.5), `oxixml-fo` (XSL-FO 1.1 tree/property model).
- Phase 7 umbrella & tools: `oxixml-rdfa` (RDFa 1.1 Core for HTML5/XHTML1/XHTML5/XML host
  languages, @role, rdfa:copy/rdfa:Pattern property copying, @datetime inference), the `oxixml`
  facade (42 feature-gated re-exports across 8 families with runnable doctests), `oxixml-cli`
  (xmllint/xsltproc-class tool: doc/fmt/lint/validate/transform/query/canon/convert/rdf-format/
  hdt/sparql subcommands with `--catalog` and `--xinclude`), and `oxixml-testsuite`
  (manifest-driven runners for every vendored conformance corpus plus an
  `oxixml-conformance-report` aggregator binary).
- Phase 8 release engineering: per-crate description/keywords/categories/readme +
  `[package.metadata.docs.rs] all-features = true` across all 44 publishable crates, per-crate
  READMEs, and `RELEASING.md` with the 8-tier topological publish order and dry-run baseline.
- Workspace scaffold: 45 layered crates, deny.toml bans for superseded upstream crates,
  workspace lints (`unsafe_code = "deny"`, `clippy::unwrap_used = "deny"`).
- `async-tokio` feature across the full RDF I/O stack (`oxixml-turtle`, `oxixml-rdfxml`,
  `oxixml-jsonld`, `oxixml-trix`, `oxixml-json`, `oxixml-io`, `oxixml-sparql-results`),
  signature-compatible with upstream oxttl/oxrdfio/sparesults, forwarded by the facade.
- `arbitrary` feature on `oxixml-xml` (Arbitrary impls for all event types, mirroring
  quick-xml 0.41) forwarded by `oxixml-quickxml-compat`; `serde` feature on `oxixml-iri`.
- Configurable element-depth guard in `oxixml-xml` (`Config::max_element_depth`,
  default 1024) surfaced through `oxixml-dom`'s `BuilderOptions::with_max_element_depth`.
- New embedding controls in `oxixml-xslt`: `Parameters::with_recursion_limit` (raises the
  default 1000-level template-instantiation ceiling) and `StaticOptions::with_backwards_compatibility`
  (default permissive; set `false` for strict `XTDE0160` rejection of evaluated `version="1.0"`
  elements); `CompiledStylesheet::transform_optional` transforms with an absent global context
  item (no source document required). `oxixml-xpath` gained `StaticContext::with_xslt_functions`
  (admits XSLT 1.0's `format-number`/`generate-id` under an XPath 1.0 static context) and an
  exported `calling_focus`, giving a host-defined function — XSLT's `fn:key`/`fn:id` — the
  calling context item.

### Fixed

- Production-hardening sweep (100+ confirmed findings across the workspace):
  XML 1.1 RestrictedChar enforcement and Namespaces 1.1 prefix undeclaration; entity-expansion
  budget (10k expansions / 10 MiB); namespace nesting-depth overflow (u16 → usize with
  checked push); UTF-16 byte-order evidence no longer overridden by a bare declared label;
  DOM end-of-line normalization, duplicate-attribute rejection, conflicting-prefix fixup and
  content-outside-root rejection; SAX ill-formedness now fatal (no infinite loop); HTML
  `<template>` contents serialization; iterative rewrites and depth guards eliminating
  stack-overflow aborts across infoset/html/xpath/xquery/jsonld/sparql/regex/dsig;
  SPARQL fresh-name collision and aggregate-serialization fixes with VALUES
  duplicate-variable rejection; F&O implicit-timezone semantics threaded end-to-end
  (XPath comparisons, XQuery FLWOR `order by`/`group by`, fn:min/max/distinct-values/
  index-of/sort/deep-equal, xsl:sort); collation-aware XSLT keys/sort/grouping;
  `TextPosition` unified on 0-based storage with 1-based Display (upstream-compatible).
- 99%-gate conformance campaign: reader end-of-input
  well-formedness checks now run exactly once (previously an unclosed document made
  `check_well_formedness` synthesize `MissingEndTag` forever — unbounded memory growth);
  `XdmMap` rewritten as a persistent hash array-mapped trie (map:put/map:remove were
  O(n) full copies, quadratic in derivation loops), with `same_key`/`same_key_hash` using
  exact-value comparison (not numeric promotion) and per-type timezone distinction; XDM
  nodes retain the written namespace prefix; xsl:context-item, xsl:map/xsl:map-entry,
  fn:current-output-uri, fn:copy-of/fn:snapshot implemented; ordering comparisons
  reject non-ordered atomic types per F&O §8.2; `castable as` follows erratum XP.E5;
  wildcard name-tests are single terminals (no interior whitespace/comments); JSON
  function family (fallback option, duplicates policies, FOJS codes) and
  format-date/dateTime/number picture handling completed; gMonth pre-erratum leniency
  dropped; CTA/assertion inheritable-attribute visibility, targetNamespace on local
  declarations, open-content extension merging and XSD 1.1 particle-restriction
  relaxations; QT3/xslt30 drivers gained a 30 s per-case budget (timeouts are recorded
  failures, never silent skips); the CLI lint two-phase workaround was removed.

### Conformance

All suites run against vendored W3C / RDF-tests corpora (discovered via `OXIXML_XML_SUITES_DIR`
and `OXIXML_TESTSUITE_DIR`; skipped silently when unset):

- W3C XML Conformance (incl. XML 1.1 + Namespaces 1.1): 984/988 not-well-formed cases
  detected (4 documented allowlist entries), 478/478 valid + invalid documents parsed;
  workspace harness well-formedness 1257/1257 and validity 931/931.
- RDF 1.1 Turtle 313/313, TriG 356/356, N-Triples 70/70, N-Quads 87/87;
  RDF 1.2 Turtle 96/96, TriG 60/60, N-Triples 70/70, N-Quads 68/68.
- RDF/XML 166/166 · RDFC-1.0 canonicalization 86/86.
- JSON-LD 1.1: expand 376/376, compact 244/244, flatten 55/55, toRdf 455/455,
  streaming toRdf 471/471, fromRdf 52/52.
- SPARQL syntax (parse + algebra round-trip, positive + negative tests all passing):
  1.0 199/199, 1.1 169/169, 1.2 198/198 (workspace harness; per-crate gate agrees).
- ISO Schematron (schematron.com corpus): 307/307 (38 suite-defective fixtures
  documented and skipped: dangling sch:extends, non-castable anyType targets).
- W3C xmlconf DTD validation (oxixml-dtd): 933/933 (incl. the TESTSUITE-rooted eduni
  errata/namespaces/XML-1.1 catalogs).
- XML Schema Test Suite: XSD 1.0 39020 pass / 118 fail (99.699%); XSD 1.1 whole-suite
  41324 pass / 127 fail (99.694%) with assertions/CTA evaluated, inheritable attributes,
  targetNamespace on local declarations, open-content extension merging and 1.1
  particle-restriction relaxations — 1.1 contribution sets (Saxonica/IBM/Oracle)
  2246 pass / 21 fail (99.074%). The single contrib case that flips to fail
  (`s3_4_2_4si03`) is a stale suite-defect: its "invalid" expectation contradicts the
  same W3C bug-15795 resolution that makes the circular attribute-group cases valid;
  contrib still clears the 99% gate.
- QT3 (30 s per-case budget): XPath 21597 passed of 21645 attempted (99.8%); XQuery
  29148 passed of 29271 attempted (99.6%). Version-pinned cases (XQ10/XQ30-exact) are
  skipped, not attempted.
- xslt30-test, 72-set core-language allowlist: 756 passed, 0 failed. Cases outside the
  allowlist's scope are skipped (streaming, packaging, schema-awareness, higher-order/
  dynamic evaluation — documented gaps), as are two cases (function-0701,
  call-template-1003) that the engine transforms correctly but whose catalog
  assertions use absolute paths the harness's boolean-assertion evaluator does not
  bind a context item for.
- html5lib-tests (oxixml-html): tokenizer 7032/7032 cases (100%) across all six initial
  states; tree construction 1792/1792 blocks (100%, scripting off, incl. the
  `<selectedcontent>` customizable-select reaction); encoding 82/82 runnable blocks.
- W3C RDFa Test Suite (rdfa1.1, via `OXIXML_RDFA_SUITE_DIR`): html5 170/170, xhtml1 181/181,
  xhtml5 177/177, xml 126/126 — 654/654 (100%) across all four host languages, verified by both
  the in-crate harness and the oxixml-testsuite runner.
- RELAX NG spectest: schemas 385/385, valid 289/289, invalid 291/291; compact syntax 87/87.
- Canonical XML 2.0 test cases: 30/30 · XMLDSig 1.1 interop: 45 documents verified as
  expected, incl. forged-signature / tampered-payload / wrong-HMAC negative sweeps.
