# oxixml

The COOLJAPAN Pure-Rust foundation for XML and RDF processing — one umbrella crate over 42
independently publishable crates in the [OxiXML](https://github.com/cool-japan/oxixml)
workspace, each hidden behind its own Cargo feature so a dependent pays only for what it asks
for. At the XML layer it is a drop-in [`quick-xml`](https://docs.rs/quick-xml) replacement (the
`xml` feature, plus a `quickxml-compat` shim for a one-line `Cargo.toml` migration); at the RDF
layer it replaces the [Oxigraph](https://oxigraph.org) toolchain (`oxrdf`, `oxttl`, `oxrdfxml`,
`oxjsonld`, `oxrdfio`, `spargebra`, `sparesults`, `oxsdatatypes`, `oxiri`, `rdf-canon`); and with
its tree, validation, query, transform and security tiers it covers the ground a JVM stack spans
with Apache Xerces, Xalan/Saxon and Apache Santuario — all of it in Pure Rust, with no C, C++ or
Fortran in any feature's default build.

## Install

There is no default feature — enable exactly what you need:

```toml
[dependencies]
oxixml = { version = "0.1.2", features = ["xml", "dom"] }

# or reach for everything at once:
# oxixml = { version = "0.1.2", features = ["full"] }
```

## Usage

```rust
use oxixml::xml::events::Event;
use oxixml::xml::reader::Reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = Reader::from_str("<catalog><book id=\"b1\"/></catalog>");
    reader.config_mut().trim_text(true);

    let mut saw_book = false;
    loop {
        match reader.read_event()? {
            Event::Empty(start) => saw_book = start.name().as_ref() == b"book",
            Event::Eof => break,
            _ => {}
        }
    }
    assert!(saw_book);
    Ok(())
}
```

Enabling a feature also enables whatever it is built on — `xslt` also reaches `xpath` and `dom`
(hence `xml`); `rdfa` also reaches `html` (hence `dom`) and `rdf`; and so on. See the [crate
documentation](https://docs.rs/oxixml) for the full feature table and one runnable example per
tier (XML reader, DOM, XPath, XSD, XSLT, Turtle, RDF canonicalization, RDF format conversion).

## What it replaces

`quick-xml` for XML parsing/writing; the Oxigraph toolchain (`oxrdf`/`oxttl`/`oxrdfxml`/
`oxjsonld`/`oxrdfio`/`spargebra`/`sparesults`/`oxsdatatypes`/`oxiri`/`rdf-canon`) for RDF and
SPARQL; and, across the validation/query/security tiers, the roles Apache Xerces, Xalan/Saxon
and Apache Santuario play in a JVM stack.

## Conformance

Each feature is backed by its own crate, gated against its own W3C or oxigraph conformance
suite where one applies — see that crate's own README for its figures. This umbrella crate
re-exports them unchanged; it adds no processing logic of its own to measure.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
