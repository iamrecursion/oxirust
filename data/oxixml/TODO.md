# OxiXML — Production Implementation TODO

**Mission:** the definitive, self-contained **Pure-Rust** home for *all* XML and RDF/Semantic-Web
processing in COOLJAPAN — present and future. Natively: Apache **Xerces + Xalan + FOP** +
**Saxon** + **Oxigraph**, in one workspace. **Foundation-pure:** 0 external crates in default
features (serde/tokio only behind off-by-default features). Budget ~200–300k LOC — 0.1.0 shipped
at 296k lines of Rust, 0.1.1 stands at 396k, 0.1.2 stands at 397k.

**Key decisions:** bundle the Xerces/Xalan tier from the start · spec depth = 1.0→3.x staircase ·
lead the processing tier with DOM+XPath+c14n · OxiXML owns syntax/model/validation, execution
engines (SPARQL exec, SHACL, FO→PDF, reasoning) stay in oxirs-arq / oxirs-shacl / fop / oxirs.

**Conformance oracle:** the vendored Oxigraph testsuite + W3C suites (xmlts / XSTS / QT3 / XSLT3 /
RELAX-NG / Schematron / c14n / xmldsig). Harness discovers them via `OXIXML_TESTSUITE_DIR` /
`OXIXML_XML_SUITES_DIR` env vars — never hardcoded paths.

**Compatibility targets:** `oxixml-xml` mirrors **quick-xml 0.41**; `oxixml-quickxml-compat`
re-exports as `quick_xml::*`. `oxixml-iri` mirrors **oxiri**, `oxixml-langtag` mirrors
**oxilangtag**, `oxixml-xsd` mirrors **oxsdatatypes**, `oxixml-model` mirrors **oxrdf**,
`oxixml-turtle` mirrors **oxttl**, `oxixml-json` mirrors **json-event-parser**,
`oxixml-sparql-results` mirrors **sparesults**, `oxixml-sparql-syntax` mirrors **spargebra**.

---

## Phase 0 — Scaffold  ✅

- [x] Virtual workspace (resolver 3, edition 2024, rust-version 1.85), 46 crate skeletons (45 at 0.1.0; `oxixml-svg` added in 0.1.1).
- [x] Workspace package/deps/lints (`unsafe_code = "deny"`, `clippy::unwrap_used = "deny"`).
- [x] `deny.toml` bans (quick-xml + Oxigraph stack → oxixml-*); umbrella facade; CLI; testsuite.
- [x] All gates green (check / clippy / deny). LICENSE present.
- [x] Workspace deps for optional features: serde 1.0.229, tokio 1.53.1 (off by default).

## Phase 1 — Foundation primitives + atoms  ✅

- [x] `oxixml-support` — SWAR byte-scan, TextPosition, SplitMix64/Pcg32, SHA-1/2 + HMAC (FIPS/RFC vectors), BigUint modpow (for RSA verify).
- [x] `oxixml-unicode` — UCD-generated tables (gc/blocks/scripts/case-fold), XML 1.0/1.1 name classes, NFC/NFD/NFKC/NFKD + NFC quick-check, NormalizationTest subset gate.
- [x] `oxixml-encoding` — UTF-8/16LE/BE, Latin-1, windows-1252, ASCII; incremental chunk-safe Decoder; encoder path; BOM + XML Appendix-F sniffing.
- [x] `oxixml-text` — base64 (strict+forgiving), hex, percent-sets, ECMAScript number formatting, XSD canonical double/float, ws collapse.
- [x] `oxixml-regex` — XSD Appendix-G + XPath F&O 5.6 flavors; Pike VM + budgeted backtracker (backrefs); class subtraction; \p category/block; FORX error mapping.
- [x] `oxixml-iri` — RFC 3987/3986, oxiri-shaped API; §5.4 vectors; relativize/resolve round-trip.
- [x] `oxixml-langtag` — RFC 5646 well-formedness + case canonicalization, oxilangtag-shaped.
- [x] `oxixml-xsd` — all XSD 1.1 datatypes, canonical forms, partial orders, checked arithmetic (Appendix E), oxsdatatypes-shaped.
- [x] `oxixml-qname` — NCName/QName validation, ExpandedName, NamespaceStack (1.0/1.1 undeclare, reserved prefixes).
- [x] Fix pass + workspace gates (check/clippy/test/deny/doc -D warnings).

## Phase 2 — XML core (quick-xml 0.41 replacement)  ✅

- [x] `oxixml-xml` core: Event/BytesStart/BytesText/… , escape/unescape, Attributes iterator (dup detection), Config, errors.
- [x] Reader: `from_str`/`from_reader`, `read_event`/`read_event_into`, `read_to_end`, `read_text`, check_end_names, trim modes, position reporting.
- [x] NsReader: prefix resolution (`resolve_element`/`resolve_attribute`, `read_resolved_event`), Writer: `write_event`, `create_element` builder, indentation, `write_bom`.
- [x] `serialize` feature: serde Deserializer (`@attr`, `$text`, `$value`, enums, seqs, overlapping-element buffering) + Serializer (`with_root`, indent) — gate on the 5 ecosystem serde consumers' patterns.
- [x] `async-tokio` feature: `read_event_into_async`, async Writer.
- [x] `encoding` feature wiring to oxixml-encoding (non-UTF-8 input).
- [x] `oxixml-quickxml-compat`: `quick_xml::*` re-export shim + smoke tests replicating quick-xml doc examples.
- [x] Gate: W3C xmlts (not-wf 844/845, valid 413/413 applicable to a non-validating parser).

## Phase 3 — RDF base (parallel with Phase 2 where deps allow)  ✅ (rdfa → Phase 7)

- [x] `oxixml-model` — oxrdf-shaped: NamedNode/BlankNode/Literal/Variable, Triple/Quad + *Ref, RDF 1.2/star triple terms, Graph/Dataset with indexes, N-Triples Display, vocab consts.
- [x] `oxixml-json` — json-event-parser-shaped streaming JSON reader/writer (SAX-style events, position errors).
- [x] `oxixml-vocab` — rdf/rdfs/xsd/owl/skos/dcterms/foaf/geosparql/sh namespaces as typed consts.
- [x] `oxixml-turtle` — oxttl-shaped: N-Triples/N-Quads/Turtle/TriG (+ N3 subset) streaming parsers + serializers, prefix events, error recovery mode. Gate: rdf-tests 1.1+1.2 all green.
- [x] `oxixml-canon` — RDFC-1.0 (URDNA2015) via support::sha. Gate: rdf-canon 86/86.
- [x] `oxixml-hdt` — HDT binary read (+write): dictionary, bitmap triples. Gate: sample HDT corpora.
- [x] `oxixml-sparql-syntax` — SPARQL 1.0/1.1/1.2 query+update parser → spargebra-shaped algebra.
- [x] `oxixml-rdfxml` — RDF/XML parse+serialize on oxixml-xml. Gate: rdf-tests rdf-xml 166/166.
- [x] `oxixml-trix` — TriX parse/serialize.
- [x] `oxixml-jsonld` — JSON-LD 1.1 expansion/compaction/flattening + toRdf/fromRdf, streaming profile, remote-context loader trait (no network in default). Gate: json-ld-api + streaming, all sections green.
- [x] `oxixml-sparql-results` — sparesults-shaped XML/JSON/CSV/TSV readers+writers. Gate: oxigraph-tests.
- [x] `oxixml-io` — oxrdfio-shaped unified parse/serialize with format autodetect.
- [x] `oxixml-rdfa` — RDFa 1.1 Core extraction (HTML5/XHTML1/XHTML5/XML hosts, @role, rdfa:copy patterns, @datetime). Gate: W3C RDFa Test Suite (rdfa1.1) 654/654 (100%) across all four host languages via OXIXML_RDFA_SUITE_DIR.

## Phase 4 — Trees & validation  ✅

- [x] `oxixml-dom` — DOM Level 3 Core-class arena tree: Document/Element/Attr/Text/Comment/PI/CDATA/DocType, namespaces, sibling/child navigation, mutation, serialization round-trip. Gate: xmlts valid 383 round-tripped.
- [x] `oxixml-sax` — SAX2-style push ContentHandler/ErrorHandler driven by oxixml-xml.
- [x] `oxixml-infoset` — XDM: node kinds, typed values, document order, XPath data model over DOM (+ streaming adapter hooks).
- [x] `oxixml-html` — HTML5 tokenizer (all states) + tree construction (core insertion modes) → DOM; fragment parsing for RDFa. Gate: html5lib-tests — tokenizer 7032/7032 (100%), tree construction 1792/1792 (100%, incl. `<selectedcontent>`), encoding 82/82 (all labels supported).
- [x] `oxixml-dtd` — internal/external subset parsing, entity expansion (billion-laughs limits), attribute defaulting, ID/IDREF, validation. Gate: xmlconf DTD 933/933 (incl. TESTSUITE-rooted eduni catalogs).
- [x] `oxixml-catalog` — OASIS XML Catalogs 1.1 resolve (public/system/rewrite/delegate).
- [x] `oxixml-xinclude` — XInclude 1.0/1.1 with fallback, xpointer element() scheme.
- [x] `oxixml-schema` — XSD: schema-doc parsing → components; validation engine: particles (UPA), complex/simple types, facets (via -regex/-xsd), identity constraints, PSVI basics; 1.1 assertions + CTA evaluated via oxixml-xpath, 1.1 xs:all restriction rules. Gate: XSTS via the workspace driver — 1.0 39271/39309 passed (38 failed, 11 skipped, 99.9%), 1.1 41434/41463 passed (29 failed, 11 skipped, 99.9%). The crate's own `tests/xsts.rs` measures the same thing independently and agrees exactly — 1.0 test sets 39116 passed / 27 failed, the 1.0-versioned groups of the 1.1 test sets 155 passed / 11 failed (39271 and 38 together), 1.1 whole suite 41434 passed / 29 failed, 1.1 contributions 2269 passed / 3 failed — and both runners compare the 67-slot residue against the expected-failure ledger in both directions.
- [x] `oxixml-relaxng` — RELAX NG full + compact syntax, Clark derivative validation. Gate: jing-trang spectest 100% (schemas 385/385, valid 289/289, invalid 291/291).
- [x] `oxixml-schematron` — ISO Schematron on XPath engine. Gate: schematron suite subset.

## Phase 5 — Query & transform (staircase 1.0 → 3.x)  ✅

- [x] `oxixml-fn` — F&O 3.1 function library over XDM values (strings/numbers/dates/sequences/regex funcs), collation framework (codepoint + UCA-lite), err:* codes.
- [x] `oxixml-xpath` — lexer/parser for 1.0→3.1 grammar (path exprs, predicates, FLWOR-lite for 3.1 `let`, maps/arrays), static context (ns/vars/functions), dynamic eval over infoset, number/string/boolean coercion rules per version.
- [x] `oxixml-xquery` — XQuery 3.1 on the xpath core: FLWOR (+ window/group by), constructors, modules, typeswitch. Gate: QT3/FOTS via the workspace driver (`testsuite/oxixml-testsuite`) — XQuery 29744/29744 passed, 0 failed, 2077 skipped (100.0%); XPath 21731/21731 passed, 0 failed, 10090 skipped (100.0%). Both denominators reconcile to the 31821-case catalogue. The run is gated on **zero wrong answers** rather than zero failures: a per-case timeout measures the machine, a wrong answer measures the processor. XQUF remains a documented gap.
- [x] `oxixml-xslt` — XSLT 1.0 complete (template rules, apply/call, keys, number, sort, output method) then 2.0/3.0 (grouping, regex instrs, functions, initial-template, xsl:context-item, xsl:map, JSON/adaptive output, `fn:transform`, `fn:current-output-uri`; streaming out of scope this cycle). Gate: xslt30-test over the **whole 14601-case catalogue** — 7340/7420 adjudicated passed, 80 failed, 7181 skipped (98.9% of adjudicated cases). All 80 failures are in the expected-failure ledger, which each run compares in both directions; 36 of them are this workspace's own backlog and are listed under "Roadmap" below. The crate's own `tests/conformance.rs` additionally guards a per-test-set floor for each area the work has concentrated on, so a regression cannot hide behind the total.

## Phase 6 — Security & formatting  ✅

- [x] `oxixml-c14n` — Canonical XML 1.0 + 1.1 + Exclusive 1.0 (+ 2.0 parse), from DOM and from streaming events; InclusiveNamespaces PrefixList. Gate: xml-c14n2 30/30, dsig-interop SignedInfo 45/45.
- [x] `oxixml-dsig` — XMLDSig Core: Reference/Transforms (enveloped, c14n, base64, XPath filter), digest via support::hash, HMAC/RSA PKCS#1 v1.5/RSASSA-PSS/DSA/ECDSA sign+verify via support::bigint, SignedInfo canonicalization; KeyInfo model; X.509 parsing + RFC 5280 §6.1 certification-path validation. Gaps: XSLT transform, EC curves beyond P-256/P-384, certificate revocation (CRL/OCSP) and policy/name-constraint checks. Gate: W3C interop 45 docs incl. forged/tampered negative sweeps.
- [x] `oxixml-enc` — XMLEnc structure model + AES-128/256-CBC/GCM (pure, safe) encrypt/decrypt, RSA-OAEP + PKCS#1 v1.5 key transport + RSA key generation, key wrap (AES-KW), ECDH-ES/DH key agreement with ConcatKDF derivation. Gaps: Triple DES + PBKDF2 (disallowed/deferred by policy), curves beyond P-256/P-384, building a `dh`/`dh-es` AgreementMethod by hand.
- [x] `oxixml-fo` — XSL-FO 1.1 object/property model, FO tree building + property resolution (inheritance, shorthands), simple area-model line-breaking primitives; **rendering stays in `fop`**.

## Phase 7 — Umbrella, CLI, testsuite harness

- [x] `oxixml` facade — all 42 re-exportable crates as features (43 with `svg` since 0.1.1; 8-family table), feature-edge audit against real API leakage, `full`, docs.rs all-features metadata, 9 runnable feature-gated doctests (the `svg` example added alongside the `svg` feature).
- [x] `oxixml-cli` — subcommands: fmt/lint/validate (xsd/rng/rnc/schematron/dtd)/transform (xslt)/query (xpath+xquery)/canon (c14n/rdf-canon)/convert (RDF + SPARQL-results format conversion)/hdt/sparql/svg (SVG-aware lint/stats/re-serialize, since 0.1.1), `--catalog`/`--xinclude` support, 180 tests incl. per-subcommand integration harness over `run_status`.
- [x] `oxixml-testsuite` — manifest runners: xmlconf, XSTS (1.0+1.1), QT3 (xpath+xquery drivers), XSLT3, RELAX NG, Schematron, c14n, xmldsig, rdf-tests, rdf-canon, JSON-LD, SPARQL, RDFa (654/654 verified against rdfa-website checkout); env-var discovery; per-suite reports + `oxixml-conformance-report` bin.

## Phase 8 — Release engineering (0.1.0 readiness)

- [x] Per-crate Cargo.toml: description/keywords/categories/readme + `[package.metadata.docs.rs] all-features = true` (all 44 publishable crates).
- [x] Per-crate README.md (what/why/example/conformance) referenced by `readme =` (Apache-2.0-only license lines, matching workspace SPDX).
- [x] `cargo doc --workspace` with `RUSTDOCFLAGS="-D warnings"` clean (both default and --all-features); examples in crate roots.
- [x] `cargo deny check` (bans + licenses + advisories + sources) clean.
- [x] `cargo publish --dry-run` per crate in topological order → RELEASING.md (8-tier Kahn order; tier-1 leaves package cleanly, rest blocked only by expected unpublished-sibling errors).
- [x] CHANGELOG.md (workspace-level).
- [x] tokei/cocomo status snapshot for docs, taken at the 0.1.0 tag: 918 Rust files, 296,319 LoC
  Rust (297,939 total SLoC); COCOMO estimate ~$10.7M / 33.9 months / 28.1 people.
  (At 0.1.1: 1,114 Rust files, 396,372 LoC Rust, 398,021 total SLoC; ~$14.5M / 38.0 months /
  33.9 people.) (At 0.1.2: 1,114 Rust files, 396,895 LoC Rust, 398,544 total SLoC; ~$14.5M /
  38.0 months / 33.9 people.)
- [x] Final stub sweep: no `todo!`/`unimplemented!`/empty pub fns anywhere (4 quick-xml-heritage FIXME comments in oxixml-xml tracked separately).
- [x] LICENSE copied into each of the 44 publishable crate dirs (verified packaged via `cargo package --list`). The crates.io `oxixml` 0.0.1 placeholder is superseded by the 0.1.0 release version.
- [x] Version bump to **0.1.0** (2026-07-29): `[workspace.package] version` + all 44 `[workspace.dependencies]` entries, README install snippets, facade/compat doc snippets, RELEASING.md baseline, CHANGELOG heading, Cargo.lock.
- [x] Release (2026-07-29): branch policy switch to `master` + release commit "Availability of 0.1.0"
  + tag `v0.1.0` + GitHub Release; all 44 publishable crates live on crates.io. Development
  continues on branch `0.1.1`.

## Phase 9 — Ecosystem migration (post-release, separate effort)

- [ ] XML consumers → oxixml-xml/compat: trivial (~11) → serde (rs3gw, legalis, oxirs-samm, oxigeo-metadata, oximedia-qc) → oxirs-core (async+NsReader+Decoder). Drop dead decls (crewland, oxigeo-qc).
- [ ] RDF consumers → oxixml-*: oxirs-core (preserve `oxiri::{Iri,IriParseError}` re-export shape), oxiephemeris, tensorlogic-oxirs-bridge.
- [ ] Ecosystem-wide `cargo deny check bans` + dependency replacement-table row (maintainer).

## Phase 10 — SVG (0.1.1)  ✅

- [x] `oxixml-svg` — typed SVG 1.1/2.0 document model over `oxixml-dom` (`SvgDocument`/`SvgElement`), reference resolution with cycle detection, CSS cascade with a documented selector subset.
- [x] Complete microsyntax layer: path data with spec error-recovery, transform lists, lengths/angles, colors (149 named), paint, viewBox/preserveAspectRatio, points, SMIL clock/timing, style declarations.
- [x] Vocabulary catalogs: 88 elements, 216 attributes, content models.
- [x] Geometry engine: bezier/arc math per SVG Appendix F, exact bounding boxes, path length/point-at-length, viewBox transforms.
- [x] `SvgBuilder` programmatic writer with correct escaping + `PathBuilder`.
- [x] SVG-aware linter (18 diagnostic codes).
- [x] `oxixml-cli` `svg` subcommand; facade `svg` feature.
- [x] Gate: W3C SVG 1.1 corpus parse/roundtrip harness green (numbers recorded in `trees/oxixml-svg/README.md`); full workspace gates green.

## Phase 11 — `oxrdf`/`oxttl` API parity (0.1.2)  ✅

- [x] `oxixml-model` — borrowed iteration: `Graph`/`Dataset` iterators and the
  `triples_for_*`/`objects_for_*`/`predicates_for_*`/`subjects_for_*` pattern queries yield
  `TripleRef`/`QuadRef`/`TermRef`/`NamedNodeRef`/`NamedOrBlankNodeRef` borrows instead of owned
  terms; the interning layer decodes by borrowing its string table, so iteration allocates
  nothing.
- [x] `oxixml-model` — borrowed mutators: `Graph::insert`, `Dataset::insert`,
  `GraphViewMut::insert` take `impl Into<TripleRef<'a>>`/`impl Into<QuadRef<'a>>`;
  `Dataset::graph_mut` takes `impl Into<GraphNameRef<'_>>`; `FromIterator`/`Extend` split into
  owned and borrowed pairs.
- [x] `oxixml-model` — `vocab::{rdf, rdfs, xsd, geosparql}` constants are `NamedNodeRef<'static>`
  (`Copy`); `Literal::datatype` returns `NamedNodeRef<'_>`; `&str` constructors (`NamedNode::new`,
  `BlankNode::new`, `Variable::new`, the `Literal` constructors) accept any lifetime via a new
  `IntoOxString` trait; `canonicalize_blank_nodes` returns `HashMap<BlankNodeRef<'_>, BlankNode>`.
  Checked against `oxrdf` 0.3.3, `oxttl` 0.2.3, `oxrdfxml` 0.2.3, `oxjsonld` 0.2.5, `oxrdfio`
  0.2.5 and `spargebra` 0.4.6; three intentional deviations from `oxrdf` are now stated in the
  crate's own documentation.
- [x] `oxixml-turtle`, `oxixml-rdfxml`, `oxixml-jsonld`, `oxixml-trix`, `oxixml-io` — every
  `serialize_triple`/`serialize_quad` takes `impl Into<TripleRef<'a>>`/`impl Into<QuadRef<'a>>`;
  existing `&owned` call sites keep compiling unchanged.
- [x] Builder options take `impl Into<String>`: `with_prefix`/`with_base_iri` on
  `TurtleParser`/`-Serializer`, `TriGParser`/`-Serializer`, `N3Parser`, `RdfXmlParser`/
  `-Serializer`, `JsonLdParser`, and `oxixml-io`'s `RdfParser`/`RdfSerializer`;
  `RdfParser::with_document_loader`'s closure bound tightened to `Send + Sync + UnwindSafe +
  RefUnwindSafe + 'static` to match `oxrdfio`.
- [x] `oxixml-jsonld` — `ReaderJsonLdParser`, `SliceJsonLdParser` and
  `TokioAsyncReaderJsonLdParser` are `Send` (document loader boxed as
  `Box<dyn DocumentLoader + Send>`).
- [x] Fixed: `JsonLdSerializer::with_prefix`/`with_base_iri` were validated and silently ignored;
  they now emit `@context`/`@base` matching `oxjsonld`'s document shape. Fixed: `oxixml-turtle`'s
  `N3Term: From<NamedNodeRef<'_>>` import now compiles in the default feature configuration
  (previously gated under `rdf-12` while the impl itself was unconditional).

---

## Roadmap — the published gap list

Every item below is a *measured* gap rather than an estimate: it is either an entry in the
expected-failure ledger the conformance runners enforce, or a skip label those runners emit with a
count. Closing a *ledgered* one requires removing its entry in the same change, or the run fails
with `N ledger entr(y/ies) name a case that no longer fails` — that is the intended cost of a table
that can be trusted.

### R1 — the XSLT 3.0 compile-time construct backlog (703 skipped cases)

`not-implemented:compile` is the largest single lever in the catalogue. The tally below is the
engine's own `GAP` line beside each skip — 704 of them, one per stylesheet the compiler declined —
so the backlog is countable rather than anecdotal. 703 of the 704 are skipped under
`not-implemented:compile`; the extra one (`result-document-1406`) reaches its gap at run time and
is skipped under `not-implemented:transform`.

| cases | construct |
|---:|---|
| 432 | `xsl:package` |
| 90 | `xsl:merge` |
| 59 | `xsl:source-document` |
| 53 | `xsl:accumulator` |
| 30 | `xsl:use-package` |
| 23 | `xsl:fork` |
| 10 | `xsl:evaluate` |
| 7 | the `parameter-document` serialization parameter |

- [ ] `xsl:package` / `xsl:use-package` — 462 of the 704, plus six ledgered failures
  (`package-001r`, `package-100`, `package-101`, `transform-005`, `transform-006`, `transform-007`).
- [ ] `xsl:merge` (90) · `xsl:source-document` (59) · `xsl:accumulator` (53) · `xsl:fork` (23) —
  the streaming-adjacent declarations, which also unblock part of the 2899 `dependency:streaming`
  cases the catalogue itself gates on.
- [ ] `xsl:evaluate` (10) and the `parameter-document` serialization parameter (7).

### R2 — the 36 named engine gaps, by what closes them

| what it needs | ids |
|---|---|
| `xsl:package` / `xsl:use-package` (R1) | `package-001r`, `package-100`, `package-101`, `transform-005`, `transform-006`, `transform-007` |
| a scope-aware rewrite of the compiler's attribute-permission table (which decides 465 passing `error` cases) | `namespace-2633`, `shadow-008`, `static-025`, `static-026` |
| preserving whitespace-only source text nodes (R4) | `id-016`, `xml-version-021` |
| the XML parser rather than the XSLT engine (R5) | `catalog-004`, `catalog-012` |
| the compile-time construct backlog itself, which these two cases *measure* | `catalog-006`, `catalog-006b` |
| a code disagreement where the reference processor raises what this one raises | `static-012`, `static-013`, `static-013c` |
| one rule each, the mechanism recorded in the ledger entry | `accessor-008`, `accessor-026`, `current-output-uri-017`, `higher-order-functions-066`, `initial-function-905`, `json-to-xml-typed-010`, `merge-085`, `namespace-6202`, `shadow-001`, `snapshot-0102a`, `snapshot-0103`, `snapshot-0103a`, `transform-004`, `use-when-0134`, `use-when-0136`, `use-when-0501`, `xml-version-041` |

Two have been narrowed to one named relation each and are the cheapest to close:

- [ ] `use-when-0136` — `element_env` in the XSLT compiler consults a node's own base URI only
  when the module has none, so a module URI shadows the base URI of content contributed by an
  external general entity and `static-base-uri()` inside it answers the wrong one
  (XSLT 3.0 §5.4.3 over XML 1.0 §4.2.2). The module already compiles; the case now fails one step
  later.
- [ ] `current-output-uri-017` — the last of XSLT 3.0 §24.3's three ids. `oxixml-xpath` builds an
  inline function as a native function item, and that arm of `Function::call` publishes no call
  site, so a host function cannot tell it is inside a function-item invocation. Needs either that
  arm to publish a dynamic call site *plus* a new any-frame predicate — **not** a change to
  `invoked_through_function_item()`, which `fn:current`, `fn:current-group` and
  `fn:current-grouping-key` depend on being innermost-only — or a public reader of the open-call
  depth.

### R3 — `cost:unicode-90`: 1460 declined cases, and the decline needs re-measuring

1460 cases (10% of the catalogue) are skipped under `cost:unicode-90`. The label is deliberately
**not** spelled `dependency:`: the catalogue declares nothing that puts these cases out of this
processor's audience, the processor answers them, and the driver declined to spend the time on a
whole-catalogue run. Naming the set lifts the decline with no code change —
`OXIXML_XSLT30_SETS=unicode-90` — and the one-line reversal is `costly_set` in
`testsuite/oxixml-testsuite/src/xslt30/mod.rs`.

- [x] Re-measured 2026-08-06, after the 156× per-call reduction, on an otherwise idle machine:
  21 cases adjudicated in ~12 minutes (~34 s per case), extrapolating to roughly **13 hours** for
  the set — the per-call fix did not move this cost, which sits inside single `fn:matches` calls
  over the 121 MB of `docs/unicode-*.xml` fixtures, exactly as predicted when the decline was
  first taken. **The decline now rests on a current measurement and the label stays.** The
  extrapolation is if anything friendly: 1444 of the 1460 cases run the per-class generator
  stylesheet, and the one of those the probe reached spent its full 30-second budget without
  finishing.
- [ ] If the set is ever admitted, investigate the probe's failure sample first: of the 21
  adjudicated, 8 passed and 13 failed, and eight of the failures are large count mismatches
  (`expected /count = 63084, got 1111262`-shaped) that look like a defect in what those cases
  measure rather than Unicode-version drift; four more are `false true` comparisons in the
  C-class, and one is the 30-second timeout above. The raw trace is reproducible with
  `OXIXML_XSLT30_SETS=unicode-90`.

### R4 — XML 1.1 and whitespace-only text nodes

- [ ] Preserve whitespace-only source text nodes (XSLT 3.0 §4.4) unless `xsl:strip-space` says
  otherwise. Closes `id-016` (ten runs of three spaces missing from the result and nothing else)
  and `xml-version-021` (one text node whose content is a LINE SEPARATOR between two spaces — XML
  1.1 normalizes U+2028 to a line feed, after which the node is whitespace-only and is dropped).
  Both resemble the suite's whitespace-defect cluster and are **not** suite defects: the reference
  processor records a plain `result="pass"` for each with no normalization comment. This touches
  the code path 7000+ passing cases run through, which is why it wants a change of its own rather
  than a corner of another one.
- [ ] `xml-version-041` — undeclaring a prefixed namespace binding with an empty URI requires
  Namespaces in XML 1.1. The diagnosis is right and the code is wrong: `XTSE0165` is raised where
  `XTSE0020` is expected, on a stylesheet the parser is entitled to reject.

### R5 — two ids away from the XML parser (`oxixml-infoset`)

`catalog-004` and `catalog-012` retrieve every stylesheet module in the suite with `fn:document`,
so one unparseable module fails them. The base-URI resolution they were once blamed on is correct;
two parser gaps remain.

- [ ] Expand external **parameter** entities: `<!ENTITY % ext SYSTEM "entity.ent">%ext;` in a
  document's internal subset is not expanded, so `whitespace-011.xsl` fails to parse.
- [ ] Give `ExternalEntity` the base URI of its declaration. It carries `name`, `public_id` and
  `system_id` and no base URI, so a resolver cannot chain: an external subset's own system
  identifiers are relative to that subset rather than to the document (XML 1.0 §4.2.2), which is
  what `include-0101b.xsl` needs. This is a breaking change to `oxixml-infoset`'s public type.

### R6 — ledger entries that must **not** be "fixed"

Recorded here so that nobody closes them by accident. Every one carries the same reasoning in the
runner's own documentation, beside the entry:

- `regex-classes-111` (`IsPrivateUse`) is a **conflict between two W3C suites**, not Unicode drift:
  the XSLT 3.0 reference results require the block to be the union of U+E000–U+F8FF,
  U+F0000–U+FFFFD and U+100000–U+10FFFD, while the XML Schema suite carries four cases asserting
  the opposite for the same escape — all four of which pass today. Extending the block closes one
  case and breaks four.
- `base-uri-052` needs an `xml:base` already present on an included document's root to survive
  XInclude processing. XInclude 1.0 §4.5.5 replaces it, which is what this workspace does, and the
  catalogue's own inline comment beside the assertion says the specification requires the
  replacement "but Xerces doesn't appear to do this".
- XML Schema `Attribute_w3c/attP031` — satisfying its 1.0 instance expectation needs a rule under
  which a `fixed` value constraint makes a `use="prohibited"` attribute use non-prohibiting. XSD
  1.0 maps `use` to `{required}` only, and the same catalogue pins the opposite three times over
  (`attF001`, `attJ002`, `attJ003`), all of which this engine passes.
- XML Schema `introspection` (graded once per pass) is a retrieval wall, not engine work:
  `common/xsts.xsd` imports XLink from an absolute `http:` URI. Measured with a stand-in XLink
  schema in place, `xsts.xsd` assembles and all 91 instance documents the group names validate.
  Do not fix it by relaxing resolution.

The XML Schema residue has **no engine gaps at all** — 65 suite defects and 2 retrieval
dependencies across its 67 slots. Four of those ids were once classed as engine gaps, were
investigated against the suite sources and the running engine, and were reclassified with no code
change.

---
*Documented scope gaps (feature-gated or deferred, stated in crate docs):* XSLT 3.0 streaming ·
XSD 1.1 full assertion coverage · XMLDSig XSLT transform + certificate-revocation checking
(CRL/OCSP) · XMLEnc Triple DES/PBKDF2 (policy-excluded) + dh/dh-es AgreementMethod construction ·
CJK legacy encodings · HDT write optimization · RDFa full suite.
