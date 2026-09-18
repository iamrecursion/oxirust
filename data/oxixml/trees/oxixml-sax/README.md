# oxixml-sax

SAX2-style push and streaming parser interfaces over `oxixml-xml`, for the
[OxiXML](https://github.com/cool-japan/oxixml) workspace.

This is a Rust translation of the classic [SAX2](https://www.saxproject.org/) event model:
instead of pulling events one at a time, `XMLReader::parse` pumps the whole document and pushes
each event to whichever handler traits you install — `ContentHandler` (element structure,
character data, namespace scope, PIs), `LexicalHandler` (comments, CDATA boundaries, the
doctype), `DtdHandler` (`NOTATION`/`NDATA`), `DeclHandler` (`ELEMENT`/`ATTLIST`/`ENTITY`) and
`ErrorHandler` (parse errors with a `Locator` snapshot), mirroring `org.xml.sax`'s traits of the
same names. Every method has a no-op default, so a handler only overrides what it needs, and
handlers are borrowed for one `parse` call rather than held as reader state — friendlier to the
borrow checker than the Java setter-based `XMLReader`.

## Install

```toml
[dependencies]
oxixml-sax = "0.1.2"
```

## Usage

```rust
use oxixml_sax::{Attributes, ContentHandler, Handlers, SaxError, XMLReader};

#[derive(Default)]
struct CountElements(u32);

impl ContentHandler for CountElements {
    fn start_element(
        &mut self,
        _uri: Option<&str>,
        _local_name: &str,
        _qname: &str,
        _attributes: &Attributes,
    ) -> Result<(), SaxError> {
        self.0 += 1;
        Ok(())
    }
}

fn main() -> Result<(), SaxError> {
    let mut content = CountElements::default();
    let mut handlers = Handlers::new().with_content(&mut content);
    XMLReader::new().parse_str("<a><b/><b/></a>", &mut handlers)?;
    assert_eq!(content.0, 3);
    Ok(())
}
```

## License

Licensed under the Apache License, Version 2.0
([LICENSE](https://github.com/cool-japan/oxixml/blob/main/LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
