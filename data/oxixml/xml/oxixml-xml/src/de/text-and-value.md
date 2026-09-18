## `$text`
`$text` is used when you want to write your XML as a text or a CDATA content.
More formally, field with that name represents simple type definition with
`{variety} = atomic` or `{variety} = union` whose basic members are all atomic,
as described in the [specification].

As a result, not all types of such fields can be serialized. Only serialization
of following types are supported:
- all primitive types (strings, numbers, booleans)
- unit variants of enumerations (serializes to a name of a variant)
- newtypes (delegates serialization to inner type)
- [`Option`] of above (`None` serializes to nothing)
- sequences (including tuples and tuple variants of enumerations) of above,
  excluding `None` and empty string elements (because it will not be possible
  to deserialize them back). The elements are separated by space(s)
- unit type `()` and unit structs (serializes to nothing)

Complex types, such as structs and maps, are not supported in this field.
If you want them, you should use `$value`.

Sequences serialized to a space-delimited string, that is why only certain
types are allowed in this mode:

```
# use serde::Deserialize;
# use oxixml_xml::de::from_str;
#[derive(Deserialize, PartialEq, Debug)]
struct AnyName {
    #[serde(rename = "$text")]
    field: Vec<usize>,
}

// A serializer would write this value as `<AnyName>1 2 3</AnyName>`
let xml = "<AnyName>1 2 3</AnyName>";

let object: AnyName = from_str(xml).expect("should be Ok");
assert_eq!(object, AnyName { field: vec![1, 2, 3] });
```

## `$value`
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: a name `#content` would better explain the purpose of that field,
but `$value` is used for compatibility with other XML serde crates, which
uses that name. This will allow you to switch XML crates more smoothly if required.
</div>

The representation of primitive types in `$value` does not differ from their
representation in `$text` fields. The difference is how sequences are serialized
and deserialized. `$value` serializes each sequence item as a separate XML element.
How the name of the XML element is chosen depends on the field's type. For
`enum`s, the variant name is used. For `struct`s, the name of the `struct`
is used.

During deserialization, if the `$value` field is an enum, then the variant's
name is matched against. That's **not** the case with structs, however, since
`serde` does not expose type names of nested fields. This does mean that **any**
type could be deserialized into a `$value` struct-type field, so long as the
struct's fields have compatible types (or are captured as text by `String`
or similar-behaving types). This can be handy when using generic types in fields
where one knows in advance what to expect. If you do not know what to expect,
however, prefer an enum with all possible variants.

Unit structs and unit type `()` serialize to nothing and can be deserialized
from any content.

Serialization and deserialization of `$value` field performed as usual, except
that name for an XML element will be given by the serialized type, instead of
field. The latter allow to serialize enumerated types, where variant is encoded
as a tag name, and, so, represent an XSD `xs:choice` schema by the Rust `enum`.

In the example below, field will be serialized as `<field/>`, because elements
get their names from the field name. It cannot be deserialized, because `Enum`
expects elements `<A/>`, `<B/>` or `<C/>`, but `AnyName` looked only for `<field/>`:

```
# use serde::{Deserialize, Serialize};
# #[derive(PartialEq, Debug)]
#[derive(Deserialize, Serialize)]
enum Enum { A, B, C }

# #[derive(PartialEq, Debug)]
#[derive(Deserialize, Serialize)]
struct AnyName {
    // <field>A</field>, <field>B</field>, or <field>C</field>
    field: Enum,
}
# assert_eq!(
#     AnyName { field: Enum::B },
#     oxixml_xml::de::from_str("<root><field>B</field></root>").expect("should be Ok"),
# );
```

If you rename field to `$value`, then `field` would be serialized as `<A/>`,
`<B/>` or `<C/>`, depending on the its content. It is also possible to
deserialize it from the same elements:

```
# use serde::{Deserialize, Serialize};
# #[derive(Deserialize, Serialize, PartialEq, Debug)]
# enum Enum { A, B, C }
#
# #[derive(PartialEq, Debug)]
#[derive(Deserialize, Serialize)]
struct AnyName {
    // <A/>, <B/> or <C/>
    #[serde(rename = "$value")]
    field: Enum,
}
# assert_eq!(
#     AnyName { field: Enum::B },
#     oxixml_xml::de::from_str("<root><B/></root>").expect("should be Ok"),
# );
```

The next example demonstrates how generic types can be used in conjunction
with `$value`-named fields to allow the reuse of wrapping structs. A common
example use case for this feature is SOAP messages, which can be commmonly
found wrapped around `<soapenv:Envelope> ... </soapenv:Envelope>`.

```rust
# use oxixml_xml::de::from_str;
# use serde::Deserialize;
#
#[derive(Deserialize, PartialEq, Debug)]
struct Envelope<T> {
    body: Body<T>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Body<T> {
    #[serde(rename = "$value")]
    inner: T,
}

// A serializer would write `Envelope { body: Body { inner: Example { a: 42 } } }`
// as `<Envelope><body><Example><a>42</a></Example></body></Envelope>` —
// notice how `inner` is not present in the XML

#[derive(Deserialize, PartialEq, Debug)]
struct AnotherExample {
    a: i32,
}

assert_eq!(
    // Notice that tag the name does nothing for struct in `$value` field
    Envelope { body: Body { inner: AnotherExample { a: 42 } } },
    from_str("<Envelope><body><Example><a>42</a></Example></body></Envelope>").expect("should be Ok"),
);
```

### Primitives and sequences of primitives

Sequences serialized to a list of elements. Note, that types that does not
produce their own tag (i. e. primitives) will produce [`SeError::Unsupported`]
if they contains more that one element, because such sequence cannot be
deserialized to the same value:

```
# use serde::Deserialize;
# use oxixml_xml::de::from_str;
#[derive(Deserialize, PartialEq, Debug)]
struct AnyName {
    #[serde(rename = "$value")]
    field: Vec<usize>,
}

// `AnyName { field: vec![1, 2, 3] }` cannot be serialized: it would be
// represented as "<AnyName>123</AnyName>", which does not round-trip

let object: AnyName = from_str("<AnyName>123</AnyName>").expect("should be Ok");
assert_eq!(object, AnyName { field: vec![123] });

// `1 2 3` is mapped to a single `usize` element
// It is impossible to deserialize list of primitives to such field
from_str::<AnyName>("<AnyName>1 2 3</AnyName>").expect_err("should be Err");
```

A particular case of that example is a string `$value` field, which probably
would be a most used example of that attribute:

```
# use serde::Deserialize;
# use oxixml_xml::de::from_str;
#[derive(Deserialize, PartialEq, Debug)]
struct AnyName {
    #[serde(rename = "$value")]
    field: String,
}

// A serializer would write this value as `<AnyName>content</AnyName>`
let object: AnyName = from_str("<AnyName>content</AnyName>").expect("should be Ok");
assert_eq!(object, AnyName { field: "content".to_string() });
```

### Structs and sequences of structs

Note, that structures do not have a serializable name as well (name of the
type is never used), so it is impossible to serialize non-unit struct or
sequence of non-unit structs in `$value` field. (sequences of) unit structs
are serialized as empty string, because units itself serializing
to nothing:

```
# use serde::Deserialize;
# use oxixml_xml::de::from_str;
#[derive(Deserialize, PartialEq, Debug)]
struct Unit;

#[derive(Deserialize, PartialEq, Debug)]
struct AnyName {
    // #[serde(default)] is required to deserialization of empty lists
    // This is a general note, not related to $value
    #[serde(rename = "$value", default)]
    field: Vec<Unit>,
}

// A serializer would write `AnyName { field: vec![Unit, Unit, Unit] }`
// as `<AnyName/>`, because units themselves serialize to nothing

let object: AnyName = from_str("<AnyName/>").expect("should be Ok");
assert_eq!(object, AnyName { field: vec![] });

let object: AnyName = from_str("<AnyName></AnyName>").expect("should be Ok");
assert_eq!(object, AnyName { field: vec![] });

let object: AnyName = from_str("<AnyName><A/><B/><C/></AnyName>").expect("should be Ok");
assert_eq!(object, AnyName { field: vec![Unit, Unit, Unit] });
```

### Enums and sequences of enums

Enumerations uses the variant name as an element name:

```
# use serde::Deserialize;
# use oxixml_xml::de::from_str;
#[derive(Deserialize, PartialEq, Debug)]
struct AnyName {
    #[serde(rename = "$value")]
    field: Vec<Enum>,
}

#[derive(Deserialize, PartialEq, Debug)]
enum Enum { A, B, C }

// A serializer would write this value using the variant names as element names
let xml = "<AnyName>\
        <A/>\
        <B/>\
        <C/>\
     </AnyName>";

let object: AnyName = from_str(xml).expect("should be Ok");
assert_eq!(object, AnyName { field: vec![Enum::A, Enum::B, Enum::C] });
```



Frequently Used Patterns
========================

Some XML constructs used so frequent, that it is worth to document the recommended
way to represent them in the Rust. The sections below describes them.

`<element>` lists
-----------------
Many XML formats wrap lists of elements in the additional container,
although this is not required by the XML rules:

```xml
<root>
  <field1/>
  <field2/>
  <list><!-- Container -->
    <element/>
    <element/>
    <element/>
  </list>
  <field3/>
</root>
```
In this case, there is a great desire to describe this XML in this way:
```
/// Represents <element/>
type Element = ();

/// Represents <root>...</root>
struct AnyName {
    // Incorrect
    list: Vec<Element>,
}
```
This will not work, because potentially `<list>` element can have attributes
and other elements inside. You should define the struct for the `<list>`
explicitly, as you do that in the XSD for that XML:
```
/// Represents <element/>
type Element = ();

/// Represents <root>...</root>
struct AnyName {
    // Correct
    list: List,
}
/// Represents <list>...</list>
struct List {
    element: Vec<Element>,
}
```

If you want to simplify your API, you could write a simple function for unwrapping
inner list and apply it via [`deserialize_with`]:

```
use oxixml_xml::de::from_str;
use serde::{Deserialize, Deserializer};

/// Represents <element/>
type Element = ();

/// Represents <root>...</root>
#[derive(Deserialize, Debug, PartialEq)]
struct AnyName {
    #[serde(deserialize_with = "unwrap_list")]
    list: Vec<Element>,
}

fn unwrap_list<'de, D>(deserializer: D) -> Result<Vec<Element>, D::Error>
where
    D: Deserializer<'de>,
{
    /// Represents <list>...</list>
    #[derive(Deserialize)]
    struct List {
        // default allows empty list
        #[serde(default)]
        element: Vec<Element>,
    }
    Ok(List::deserialize(deserializer)?.element)
}

assert_eq!(
    AnyName { list: vec![(), (), ()] },
    from_str("
        <root>
          <list>
            <element/>
            <element/>
            <element/>
          </list>
        </root>
    ").expect("should be Ok"),
);
```

Instead of writing such functions manually, you also could try <https://lib.rs/crates/serde-query>.

Overlapped (Out-of-Order) Elements
----------------------------------
In the case that the list might contain tags that are overlapped with
tags that do not correspond to the list (this is a usual case in XML
documents) like this:
```xml
<any-name>
  <item/>
  <another-item/>
  <item/>
  <item/>
</any-name>
```
you should enable the [`overlapped-lists`] feature to make it possible
to deserialize this to:
```no_run
# use serde::Deserialize;
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct AnyName {
    item: Vec<()>,
    another_item: (),
}
```


Internally Tagged Enums
-----------------------
[Tagged enums] are currently not supported because of an issue in the Serde
design (see [serde#1183] and [quick-xml#586]) and missing optimizations in
Serde which could be useful for XML parsing ([serde#1495]). This can be worked
around by manually implementing deserialize with `#[serde(deserialize_with = "func")]`
or implementing [`Deserialize`](serde::Deserialize), but this can get very tedious very fast for
files with large amounts of tagged enums. To help with this issue quick-xml
provides a macro [`impl_deserialize_for_internally_tagged_enum!`]. See the
macro documentation for details.


[`overlapped-lists`]: ../index.html#overlapped-lists
[specification]: https://www.w3.org/TR/xmlschema11-1/#Simple_Type_Definition
[`deserialize_with`]: https://serde.rs/field-attrs.html#deserialize_with
[`xsi:nil`]: https://www.w3.org/TR/xmlschema-1/#xsi_nil
[`Serializer::serialize_unit_variant`]: serde::Serializer::serialize_unit_variant
[`Deserializer::deserialize_enum`]: serde::Deserializer::deserialize_enum
[`SeError::Unsupported`]: crate::errors::serialize::SeError::Unsupported
[Tagged enums]: https://serde.rs/enum-representations.html#internally-tagged
[serde#1183]: https://github.com/serde-rs/serde/issues/1183
[serde#1495]: https://github.com/serde-rs/serde/issues/1495
[quick-xml#586]: https://github.com/tafia/quick-xml/issues/586
[`impl_deserialize_for_internally_tagged_enum!`]: https://docs.rs/quick-xml/0.41.0/quick_xml/macro.impl_deserialize_for_internally_tagged_enum.html
