# OxiXML

**The Pure-Rust foundation for all XML and RDF processing in the COOLJAPAN ecosystem.**

OxiXML is a single, foundation-pure workspace that natively implements the capabilities of
Apache **Xerces** + **Xalan** + **FOP**, **Saxon**, and **Oxigraph** — from a byte-level XML
scanner up through XSLT 3.0 / XQuery / XSD 1.1 / RELAX NG / Schematron / XMLDSig / XSL-FO, and the
complete RDF 1.2 / Turtle / RDF-XML / JSON-LD / RDFa / canonicalization / SPARQL-results stack.

- **Pure Rust, 0 external crates in default features.** Optional features only (`serialize`,
  `async-tokio`, `encoding`, `simd`, `crypto`). Depends on nothing else in the ecosystem, so any
  project can rely on it with no dependency cycles.
- **Drop-in migration.** `oxixml-xml` mirrors the `quick-xml` 0.41 API (`oxixml-quickxml-compat`
  re-exports it under `quick_xml::*`); `oxixml-model` is oxrdf-shaped.
- **Conformance-driven.** Every parser, validator, and engine is gated on its W3C test suite.

> **Status:** **0.1.3** (0.1.2 released 2026-08-10) — every
> implementation phase complete (XML core, trees/validation, query/transform,
> security/formatting, RDF/SPARQL, SVG); migrating the rest of the ecosystem onto OxiXML is a
> separate effort. 396,895 lines of Rust across 1,114 files in 46 crates; 9,775 tests and 900
> doc-tests passing with `--all-features` (7,121 and 818 with default features), 0 failures.
> 0.1.2 breaks compatibility with 0.1.1 deliberately (`oxrdf`/`oxttl` API parity in
> `oxixml-model`) — see [`CHANGELOG.md`](CHANGELOG.md). See [`TODO.md`](TODO.md) for
> phase-by-phase detail and the published gap list.

## Installation

```toml
[dependencies]
oxixml = "0.1.2"
```

Every crate in the table below is also published independently; `oxixml` is a facade that
re-exports them behind feature flags.

## Workspace layout

| Tier | Crates | Replaces / implements |
|---|---|---|
| Primitives | `oxixml-support` · `-unicode` · `-encoding` · `-text` · `-regex` | (internal foundation) |
| Atoms | `oxixml-iri` · `-langtag` · `-xsd` · `-qname` | oxiri · oxilangtag · oxsdatatypes |
| XML core | `oxixml-xml` · `-quickxml-compat` | **quick-xml 0.41** |
| Trees | `oxixml-dom` · `-sax` · `-infoset` · `-html` · `-svg` | Xerces DOM/SAX · XDM · HTML5 · SVG 1.1/2 |
| Validation | `oxixml-dtd` · `-schema` · `-relaxng` · `-schematron` · `-catalog` · `-xinclude` | Xerces XSD · Jing · Schematron |
| Query/transform | `oxixml-fn` · `-xpath` · `-xquery` · `-xslt` | Xalan / Saxon |
| Security/format | `oxixml-c14n` · `-dsig` · `-enc` · `-fo` | XMLDSig · XMLEnc · Apache FOP |
| RDF | `oxixml-model` · `-json` · `-turtle` · `-rdfxml` · `-jsonld` · `-rdfa` · `-trix` · `-hdt` · `-canon` · `-io` · `-vocab` | the Oxigraph RDF stack |
| SPARQL | `oxixml-sparql-results` · `-sparql-syntax` | sparesults · spargebra/sparopt |
| Top | `oxixml` (facade) · `oxixml-cli` · `oxixml-testsuite` | — |

Execution engines (SPARQL query execution, SHACL validation, FO→PDF rendering, OWL/RDFS reasoning)
stay in their host projects (`oxirs-arq`, `oxirs-shacl`, `fop`, `oxirs`) and depend on OxiXML for
parsing, the data model, and validation primitives.

## Conformance

Every engine is gated on its W3C test suite. The five largest, measured on the 0.1.1 tree from one
binary built once and driven repeatedly:

```text
XSLT 3.0    xslt30-test        7340 / 7420 adjudicated passed,  80 failed,  7181 skipped
                               98.9% of adjudicated cases, over the whole 14601-case catalogue
XPath 3.1   QT3/FOTS          21731 / 21731 passed,   0 failed, 10090 skipped   100.0%
XQuery 3.1  QT3/FOTS          29744 / 29744 passed,   0 failed,  2077 skipped   100.0%
XSD 1.0     xsdtests          39271 / 39309 passed,  38 failed,     11 skipped   99.9%
XSD 1.1     xsdtests          41434 / 41463 passed,  29 failed,     11 skipped   99.9%
```

**Every failure in that table is named in an expected-failure ledger inside the runner, and each
run asserts that its failing set is exactly that ledger — in both directions.** A case that starts
failing is a regression; a ledgered case that stops failing is a stale claim; both turn the run
red, which is what separates a ledger from an allowlist. An entry carries one of five classes and
its evidence: `suite-defect`, the case or its expectation is wrong or unsatisfiable, with a
specification citation; `dependency`, a capability this processor declares it does not have
(checked against the engine's own `system-property()` answer) or a resource the run cannot reach;
`unicode-version`, an expectation pinned to a Unicode version this workspace has moved past;
`performance`, a case that does not finish inside the per-case budget — **currently empty**; and
`engine-gap`, a genuine gap here, each with the mechanism that closes it, tracked as roadmap work
in [`TODO.md`](TODO.md). Of the 147 entries across the XSLT 3.0 and XML Schema suites, 111 are the
suite's own defect or a declared dependency and **36 are this workspace's backlog**. The QT3
drivers hold no entries: they are gated on zero *wrong answers* rather than zero failures, because
a per-case timeout measures the machine while a wrong answer measures the processor.

The remaining suites, as measured for the 0.1.0 release and itemised in
[`CHANGELOG.md`](CHANGELOG.md), run at 100% of their cases: RDF 1.1/1.2 Turtle · TriG ·
N-Triples · N-Quads · RDF/XML · RDFC-1.0 · JSON-LD 1.1 · SPARQL 1.0/1.1/1.2 syntax · RELAX NG ·
html5lib-tests · RDFa 1.1 · Canonical XML 2.0 · XMLDSig interop · xmlconf DTD validation. Two sit
just under 100% and say why: W3C XML conformance detects 984 of 988 not-well-formed cases (four
documented allowlist entries), and ISO Schematron is 307/307 with 38 suite-defective fixtures
documented and skipped.

Every suite is discovered from `OXIXML_XML_SUITES_DIR` / `OXIXML_TESTSUITE_DIR` and is skipped
silently when unset, so the corpora are never vendored into this repository.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
