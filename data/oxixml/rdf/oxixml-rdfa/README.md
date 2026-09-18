# oxixml-rdfa

An [RDFa 1.1](https://www.w3.org/TR/rdfa-core/) extractor for HTML, XHTML and XML, for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust foundation for
XML and RDF processing.

`RdfaParser` walks an `oxixml-dom` tree (produced by `oxixml-html` for `text/html`, or
`oxixml-xml` for XHTML/XML) applying the fourteen processing steps of [RDFa Core 1.1
§7.5](https://www.w3.org/TR/rdfa-core/#s_sequence): the full evaluation-context model, `@vocab`,
`@prefix` and legacy `xmlns:` mappings, `@typeof`/`@about`/`@resource`/`@href`/`@src`,
`@property`/`@rel`/`@rev`, `@content`/`@datatype`/`@inlist`, chained and hanging incomplete
triples, typed and `xml:lang`-tagged literals, `rdf:XMLLiteral`, RDF collections, the HTML+RDFa
1.1 extensions (`@datetime`/`<time>`, `rdf:HTML`, property copying), the published initial
contexts, and `@role` per the Role Attribute 1.0 spec.

## Install

```toml
[dependencies]
oxixml-rdfa = "0.1.2"
```

## Example

```rust
use oxixml_rdfa::RdfaParser;

let html = r##"<!DOCTYPE html>
<html prefix="schema: http://schema.org/">
  <body>
    <div about="#me" typeof="schema:Person">
      <span property="schema:name">Natsume Sōseki</span>
      <a property="schema:url" href="https://example.com/soseki">home page</a>
    </div>
  </body>
</html>"##;

let triples = RdfaParser::new()
    .with_base_iri("https://example.com/about")?
    .for_slice(html)
    .collect::<Result<Vec<_>, _>>()?;

assert_eq!(triples.len(), 3);
# Result::<_, Box<dyn std::error::Error>>::Ok(())
```

## Replaces

No direct upstream equivalent — RDFa extraction is unique to OxiXML within the COOLJAPAN
ecosystem.

## Conformance

Against the W3C [RDFa Test Suite](https://rdfa.info/test-suite/)'s `rdfa1.1` cases: `html5`
170/170, `xhtml1` 181/181, `xhtml5` 177/177, `xml` 126/126 (run via `tests/rdfa_suite.rs`, gated on
the `OXIXML_RDFA_SUITE_DIR` environment variable pointing at a checkout of
[rdfa/rdfa-website](https://github.com/rdfa/rdfa-website)). `tests/inline.rs` covers the same
ground with self-contained cases that need no suite checkout.

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
