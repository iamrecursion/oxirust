# oxixml-qname

XML names, NCName/QName, and namespace-URI handling — the name-shaped vocabulary that every
other layer of [OxiXML](https://github.com/cool-japan/oxixml) (SAX/DOM events, schema
validators, XPath/XQuery, canonicalization) builds identity on top of.

* `is_name` / `is_ncname` / `is_nmtoken` / `is_qname` — the boolean character-class predicates
  from [Namespaces in XML 1.0/1.1](https://www.w3.org/TR/xml-names/).
* `QName` — a *lexical*, unresolved qualified name (`prefix:local` or just `local`), generic
  over its text storage: parse zero-copy into `QName<&str>`, or own one via `QName<String>`.
* `ExpandedName` / `ExpandedNameRef` — the *resolved* namespace-URI-plus-local-name pair that is
  the universal name identity once a `QName` has been resolved against a namespace context;
  `Eq`/`Hash` are defined on `(namespace, local)` alone, independent of whichever prefix was
  used to spell the name, though a prefix can still be retained as an annotation via
  `with_prefix`/`prefix`.
* `NamespaceStack` — the scoped `prefix -> namespace-name` binding table a streaming parser
  pushes/pops one element at a time, implementing the reserved-prefix rules and the
  default-vs-no-namespace distinction of Namespaces in XML 1.0/1.1 §3, including the 1.1-only
  "undeclare with an empty URI" relaxation.

## Install

```sh
cargo add oxixml-qname@0.1.2
```

## Usage

```rust
use oxixml_qname::{is_ncname, is_qname, NameContext, NamespaceStack, NamespacesVersion, QName};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Character-class predicates.
    assert!(is_ncname("book"));
    assert!(!is_ncname("book:title")); // a colon is not allowed inside an NCName
    assert!(is_qname("book:title"));

    // Lexical QName parsing: zero-copy over a borrowed `&str`...
    let qname = QName::parse("book:title")?;
    assert_eq!(qname.prefix(), Some("book"));
    assert_eq!(qname.local_part(), "title");

    // A namespace stack resolves prefixes in scope, one element at a time.
    let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_1);
    namespaces.push_scope();
    namespaces.declare(Some("book"), "urn:example:book")?;
    let element_name = namespaces.resolve_qname("book:title", NameContext::Element)?;
    assert_eq!(element_name.namespace(), Some("urn:example:book"));
    namespaces.pop_scope();
    Ok(())
}
```

## Replaces

No single upstream crate — this consolidates what the Oxigraph stack spreads across ad hoc
string handling in `oxrdf`/`oxttl`/`oxrdfxml` into one dedicated, spec-exact NCName/QName/
namespace-scoping atom, generic over borrowed or owned storage the same way the sibling
`oxixml-iri` crate is.

## License

Licensed under the Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>).
See the `LICENSE` file at the root of the [OxiXML](https://github.com/cool-japan/oxixml)
workspace.
