# oxixml-fo

The XSL-FO 1.1 formatting-object tree and property model, in Pure Rust.

`oxixml-fo` is the **front half of an XSL-FO formatter**, the part Apache FOP calls the *FO
tree*: it reads a stylesheet result in the XSL-FO namespace, builds a tree of typed formatting
objects, validates it against the content models of XSL 1.1 §6, and *refines* every property of
every object per §5 — shorthand expansion, inheritance, initial values, expression evaluation,
and the writing-mode-dependent correspondence between relative and absolute property spellings.
Pagination, area-tree construction, font metrics and PDF (or any other) output are out of
scope — this crate hands a downstream renderer a fully refined `FoTree` plus the shared
area/line-breaking vocabulary in `area`.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-fo = "0.1.2"
```

## Example

```rust
use oxixml_fo::fo::FoElement;
use oxixml_fo::property::{Length, PropertyId, Value};
use oxixml_fo::FoDocument;

fn main() {
    let tree = FoDocument::parse(
        r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format" font-size="10pt">
             <fo:layout-master-set>
               <fo:simple-page-master master-name="A4"
                                      page-height="297mm" page-width="210mm">
                 <fo:region-body margin="20mm"/>
               </fo:simple-page-master>
             </fo:layout-master-set>
             <fo:page-sequence master-reference="A4">
               <fo:flow flow-name="xsl-region-body">
                 <fo:block font-size="150%" space-before="6pt" margin-left="1em">
                   Hello, formatting objects.
                 </fo:block>
               </fo:flow>
             </fo:page-sequence>
           </fo:root>"#,
    )
    .expect("a well-formed XSL-FO document");

    // Find the fo:block and inspect its refined properties.
    let block = tree
        .elements()
        .iter()
        .copied()
        .find(|id| tree.element(*id) == Some(FoElement::Block))
        .expect("the flow contains a block");
    let properties = tree.properties(block);

    // `font-size` is inherited, so XSL 1.1 §5.1.4 lets it sit on `fo:root`;
    // `150%` is 150% of the 10pt that reaches the block through `fo:flow`.
    assert_eq!(
        properties.get(PropertyId::FontSize),
        Some(&Value::Length(Length::from_points(15.0)))
    );
}
```

## What it replaces

`oxixml-fo` replaces the FO-tree front end of Apache FOP (Java) — the part of an XSL-FO
formatter upstream of pagination and rendering.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
