# oxixml-catalog

OASIS XML Catalogs 1.1 resolution, in Pure Rust.

A `Catalog` parses one OASIS XML Catalog document and answers the three lookups the
specification defines: `resolve_system` (map a `SYSTEM` identifier to a URI), `resolve_public`
(map a `PUBLIC` identifier, honoring the `prefer` search strategy), and `resolve_uri` (map an
arbitrary URI reference, such as a schema `schemaLocation` or an XInclude `href`). Every element
of the OASIS Committee Specification 1.1 is implemented, including longest-match tie-breaking,
delegate-chain restart semantics, and loop detection. The crate never touches the filesystem or
the network itself — following `nextCatalog`/`delegate*` entries to other catalog documents goes
through a caller-supplied `CatalogLoader`.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-catalog = "0.1.2"
```

## Example

```rust
use oxixml_catalog::Catalog;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="public">
      <public publicId="-//OASIS//DTD DocBook XML V4.5//EN" uri="docbook/docbookx.dtd"/>
      <rewriteSystem systemIdStartString="http://example.com/dtds/"
                     rewritePrefix="local-dtds/"/>
    </catalog>
    "#;

    let catalog = Catalog::parse(xml, "file:///etc/xml/catalog.xml")?;

    assert_eq!(
        catalog.resolve_public("-//OASIS//DTD DocBook XML V4.5//EN", None).as_deref(),
        Some("file:///etc/xml/docbook/docbookx.dtd")
    );
    assert_eq!(
        catalog.resolve_system("http://example.com/dtds/foo.dtd").as_deref(),
        Some("file:///etc/xml/local-dtds/foo.dtd")
    );
    assert_eq!(catalog.resolve_system("urn:unknown"), None);
    Ok(())
}
```

## What it replaces

There is no widely used Pure-Rust XML Catalog resolver. `oxixml-catalog` replaces the
catalog-resolution modules of libxml2 (`xmlCatalog`) or the Apache `xml-commons` resolver.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
