Serde `Deserializer` module.

Due to the complexity of the XML standard and the fact that Serde was developed
with JSON in mind, not all Serde concepts apply smoothly to XML. This leads to
that fact that some XML concepts are inexpressible in terms of Serde derives
and may require manual deserialization.

The most notable restriction is the ability to distinguish between _elements_
and _attributes_, as no other format used by serde has such a conception.

Due to that the mapping is performed in a best effort manner.



Table of Contents
=================
- [Mapping XML to Rust types](#mapping-xml-to-rust-types)
  - [Basics](#basics)
  - [Optional attributes and elements](#optional-attributes-and-elements)
  - [Choices (`xs:choice` XML Schema type)](#choices-xschoice-xml-schema-type)
  - [Sequences (`xs:all` and `xs:sequence` XML Schema types)](#sequences-xsall-and-xssequence-xml-schema-types)
- [Mapping of `xsi:nil`](#mapping-of-xsinil)
- [Generate Rust types from XML](#generate-rust-types-from-xml)
- [Composition Rules](#composition-rules)
- [Enum Representations](#enum-representations)
  - [Normal enum variant](#normal-enum-variant)
  - [`$text` enum variant](#text-enum-variant)
- [`$text` and `$value` special names](#text-and-value-special-names)
  - [`$text`](#text)
  - [`$value`](#value)
    - [Primitives and sequences of primitives](#primitives-and-sequences-of-primitives)
    - [Structs and sequences of structs](#structs-and-sequences-of-structs)
    - [Enums and sequences of enums](#enums-and-sequences-of-enums)
- [Frequently Used Patterns](#frequently-used-patterns)
  - [`<element>` lists](#element-lists)
  - [Overlapped (Out-of-Order) Elements](#overlapped-out-of-order-elements)
  - [Internally Tagged Enums](#internally-tagged-enums)



Mapping XML to Rust types
=========================

Type names are never considered when deserializing, so you can name your
types as you wish. Other general rules:
- `struct` field name could be represented in XML only as an attribute name
  or an element name;
- `enum` variant name could be represented in XML only as an attribute name
  or an element name;
- the unit struct, unit type `()` and unit enum variant can be deserialized
  from any valid XML content:
  - attribute and element names;
  - attribute and element values;
  - text or CDATA content (including mixed text and CDATA content).

<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: All tests are marked with an `ignore` option, even though they do
compile. This is  because rustdoc marks such blocks with an information
icon unlike `no_run` blocks.

</div>

<table>
<thead>
<tr><th colspan="2">

## Basics

</th></tr>
<tr><th>To parse all these XML's...</th><th>...use these Rust type(s)</th></tr>
</thead>
<tbody style="vertical-align:top;">
<tr>
<td>
Content of attributes and text / CDATA content of elements (including mixed
text and CDATA content):

```xml
<... ...="content" />
```
```xml
<...>content</...>
```
```xml
<...><![CDATA[content]]></...>
```
```xml
<...>text<![CDATA[cdata]]>text</...>
```
Mixed text / CDATA content represents one logical string, `"textcdatatext"` in that case.
</td>
<td>

You can use any type that can be deserialized from an `&str`, for example:
- [`String`] and [`&str`]
- [`Cow<str>`](std::borrow::Cow)
- [`u32`], [`f32`] and other numeric types
- `enum`s, like
  ```
  # use serde::Deserialize;
  # #[derive(Debug, PartialEq)]
  #[derive(Deserialize)]
  enum Language {
    Rust,
    Cpp,
    #[serde(other)]
    Other,
  }
  # #[derive(Debug, PartialEq, Deserialize)]
  # struct X { #[serde(rename = "$text")] x: Language }
  # assert_eq!(X { x: Language::Rust  }, oxixml_xml::de::from_str("<x>Rust</x>").expect("should be Ok"));
  # assert_eq!(X { x: Language::Cpp   }, oxixml_xml::de::from_str("<x>C<![CDATA[p]]>p</x>").expect("should be Ok"));
  # assert_eq!(X { x: Language::Other }, oxixml_xml::de::from_str("<x><![CDATA[other]]></x>").expect("should be Ok"));
  ```

<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: deserialization to non-owned types (i.e. borrow from the input),
such as `&str`, is possible only if you parse document in the UTF-8
encoding and content does not contain entity references such as `&amp;`,
or character references such as `&#xD;`, as well as text content represented
by one piece of [text] or [CDATA] element. Violating any of these conditions
does not panic: [`serde`] itself rejects the owned string it was actually
given with `serde::de::Error::invalid_type`, surfaced through this crate as
[`DeError::Custom`].
</div>

[text]: crate::events::Event::Text
[CDATA]: crate::events::Event::CData
[`DeError::Custom`]: crate::errors::serialize::DeError::Custom
</td>
</tr>
<!-- 2 ===================================================================================== -->
<tr>
<td>

Content of attributes and text / CDATA content of elements (including mixed
text and CDATA content), which represents a space-delimited lists, as
specified in the XML Schema specification for [`xs:list`] `simpleType`:

```xml
<... ...="element1 element2 ..." />
```
```xml
<...>
  element1
  element2
  ...
</...>
```
```xml
<...><![CDATA[
  element1
  element2
  ...
]]></...>
```

[`xs:list`]: https://www.w3.org/TR/xmlschema11-2/#list-datatypes
</td>
<td>

Use any type that deserialized using [`deserialize_seq()`] call, for example:

```
type List = Vec<u32>;
```

See the next row to learn where in your struct definition you should
use that type.

According to the XML Schema specification, delimiters for elements is one
or more space (`' '`, `'\r'`, `'\n'`, and `'\t'`) character(s).

<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: according to the XML Schema restrictions, you cannot escape those
white-space characters, so list elements will _never_ contain them.
In practice you will usually use `xs:list`s for lists of numbers or enumerated
values which looks like identifiers in many languages, for example, `item`,
`some_item` or `some-item`, so that shouldn't be a problem.

NOTE: according to the XML Schema specification, list elements can be
delimited only by spaces. Other delimiters (for example, commas) are not
allowed.

</div>

[`deserialize_seq()`]: serde::de::Deserializer::deserialize_seq
</td>
</tr>
<!-- 3 ===================================================================================== -->
<tr>
<td>
A typical XML with attributes. The root tag name does not matter:

```xml
<any-tag one="..." two="..."/>
```
</td>
<td>

A structure where each XML attribute is mapped to a field with a name
starting with `@`. Because Rust identifiers do not permit the `@` character,
you should use the `#[serde(rename = "@...")]` attribute to rename it.
The name of the struct itself does not matter:

```
# use serde::Deserialize;
# type T = ();
# type U = ();
// Get both attributes
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@one")]
  one: T,

  #[serde(rename = "@two")]
  two: U,
}
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..." two="..."/>"#).expect("should be Ok");
```
```
# use serde::Deserialize;
# type T = ();
// Get only the one attribute, ignore the other
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@one")]
  one: T,
}
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..." two="..."/>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."/>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."><one>...</one></any-tag>"#).expect("should be Ok");
```
```
# use serde::Deserialize;
// Ignore all attributes
// You can also use the `()` type (unit type)
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName;
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..." two="..."/>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."><one>...</one></any-tag>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag><one>...</one><two>...</two></any-tag>"#).expect("should be Ok");
```

All these structs can be used to deserialize from an XML on the
left side depending on amount of information that you want to get.
Of course, you can combine them with elements extractor structs (see below).

<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: XML allows you to have an attribute and an element with the same name
inside the one element. quick-xml deals with that by prepending a `@` prefix
to the name of attributes.
</div>
</td>
</tr>
<!-- 4 ===================================================================================== -->
<tr>
<td>
A typical XML with child elements. The root tag name does not matter:

```xml
<any-tag>
  <one>...</one>
  <two>...</two>
</any-tag>
```
</td>
<td>
A structure where each XML child element is mapped to the field.
Each element name becomes a name of field. The name of the struct itself
does not matter:

```
# use serde::Deserialize;
# type T = ();
# type U = ();
// Get both elements
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  one: T,
  two: U,
}
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag><one>...</one><two>...</two></any-tag>"#).expect("should be Ok");
#
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..." two="..."/>"#).expect_err("should be Err");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."><two>...</two></any-tag>"#).expect_err("should be Err");
```
```
# use serde::Deserialize;
# type T = ();
// Get only the one element, ignore the other
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  one: T,
}
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag><one>...</one><two>...</two></any-tag>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."><one>...</one></any-tag>"#).expect("should be Ok");
```
```
# use serde::Deserialize;
// Ignore all elements
// You can also use the `()` type (unit type)
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName;
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..." two="..."/>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag><one>...</one><two>...</two></any-tag>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."><two>...</two></any-tag>"#).expect("should be Ok");
# oxixml_xml::de::from_str::<AnyName>(r#"<any-tag one="..."><one>...</one></any-tag>"#).expect("should be Ok");
```

All these structs can be used to deserialize from an XML on the
left side depending on amount of information that you want to get.
Of course, you can combine them with attributes extractor structs (see above).

<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: XML allows you to have an attribute and an element with the same name
inside the one element. quick-xml deals with that by prepending a `@` prefix
to the name of attributes.
</div>
</td>
</tr>
<!-- 5 ===================================================================================== -->
<tr>
<td>
An XML with an attribute and a child element named equally:

```xml
<any-tag field="...">
  <field>...</field>
</any-tag>
```
</td>
<td>

You MUST specify `#[serde(rename = "@field")]` on a field that will be used
for an attribute:

```
# use serde::Deserialize;
# type T = ();
# type U = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@field")]
  attribute: T,
  field: U,
}
# assert_eq!(
#   AnyName { attribute: (), field: () },
#   oxixml_xml::de::from_str(r#"
#     <any-tag field="...">
#       <field>...</field>
#     </any-tag>
#   "#).expect("should be Ok"),
# );
```
</td>
</tr>
<!-- ======================================================================================= -->
<tr><th colspan="2">

## Optional attributes and elements

</th></tr>
<tr><th>To parse all these XML's...</th><th>...use these Rust type(s)</th></tr>
<!-- 6 ===================================================================================== -->
<tr>
<td>
An optional XML attribute that you want to capture.
The root tag name does not matter:

```xml
<any-tag optional="..."/>
```
```xml
<any-tag/>
```
</td>
<td>

A structure with an optional field, renamed according to the requirements
for attributes:

```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@optional")]
  optional: Option<T>,
}
# assert_eq!(AnyName { optional: Some(()) }, oxixml_xml::de::from_str(r#"<any-tag optional="..."/>"#).expect("should be Ok"));
# assert_eq!(AnyName { optional: None     }, oxixml_xml::de::from_str(r#"<any-tag/>"#).expect("should be Ok"));
```
When the XML attribute is present, type `T` will be deserialized from
an attribute value (which is a string). Note, that if `T = String` or other
string type, the empty attribute is mapped to a `Some("")`, whereas `None`
represents the missed attribute:
```xml
<any-tag optional="..."/><!-- Some("...") -->
<any-tag optional=""/>   <!-- Some("") -->
<any-tag/>               <!-- None -->
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: The behaviour is not symmetric by default. `None` will be serialized as
`optional=""`. This behaviour is consistent across serde crates. You should add
`#[serde(skip_serializing_if = "Option::is_none")]` attribute to the field to
skip `None`s.
</div>
</td>
</tr>
<!-- 7 ===================================================================================== -->
<tr>
<td>
An optional XML elements that you want to capture.
The root tag name does not matter:

```xml
<any-tag/>
  <optional>...</optional>
</any-tag>
```
```xml
<any-tag/>
  <optional/>
</any-tag>
```
```xml
<any-tag/>
```
</td>
<td>

A structure with an optional field:

```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  optional: Option<T>,
}
# assert_eq!(AnyName { optional: Some(()) }, oxixml_xml::de::from_str(r#"<any-tag><optional>...</optional></any-tag>"#).expect("should be Ok"));
# assert_eq!(AnyName { optional: None     }, oxixml_xml::de::from_str(r#"<any-tag/>"#).expect("should be Ok"));
```
When the XML element is present, type `T` will be deserialized from an
element (which is a string or a multi-mapping -- i.e. mapping which can have
duplicated keys).
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: The behaviour is not symmetric by default. `None` will be serialized as
`<optional/>`. This behaviour is consistent across serde crates. You should add
`#[serde(skip_serializing_if = "Option::is_none")]` attribute to the field to
skip `None`s.

NOTE: Deserializer will automatically handle a [`xsi:nil`] attribute and set field to `None`.
For more info see [Mapping of `xsi:nil`](#mapping-of-xsinil).
</div>
</td>
</tr>
<!-- ======================================================================================= -->
<tr><th colspan="2">

## Choices (`xs:choice` XML Schema type)

</th></tr>
<tr><th>To parse all these XML's...</th><th>...use these Rust type(s)</th></tr>
<!-- 8 ===================================================================================== -->
<tr>
<td>
An XML with different root tag names, as well as text / CDATA content:

```xml
<one field1="...">...</one>
```
```xml
<two>
  <field2>...</field2>
</two>
```
```xml
Text <![CDATA[or (mixed)
CDATA]]> content
```
</td>
<td>

An enum where each variant has the name of a possible root tag. The name of
the enum itself does not matter.

If you need to get the textual content, mark a variant with `#[serde(rename = "$text")]`.

All these structs can be used to deserialize from any XML on the
left side depending on amount of information that you want to get:

```
# use serde::Deserialize;
# type T = ();
# type U = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnyName {
  One { #[serde(rename = "@field1")] field1: T },
  Two { field2: U },

  /// Use unit variant, if you do not care of a content.
  /// You can use tuple variant if you want to parse
  /// textual content as an xs:list.
  /// Struct variants are will pass a string to the
  /// struct enum variant visitor, which typically
  /// returns Err(Custom)
  #[serde(rename = "$text")]
  Text(String),
}
# assert_eq!(AnyName::One { field1: () }, oxixml_xml::de::from_str(r#"<one field1="...">...</one>"#).expect("should be Ok"));
# assert_eq!(AnyName::Two { field2: () }, oxixml_xml::de::from_str(r#"<two><field2>...</field2></two>"#).expect("should be Ok"));
# assert_eq!(AnyName::Text("text  cdata ".into()), oxixml_xml::de::from_str(r#"text <![CDATA[ cdata ]]>"#).expect("should be Ok"));
```
```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct Two {
  field2: T,
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnyName {
  // `field1` content discarded
  One,
  Two(Two),
  #[serde(rename = "$text")]
  Text,
}
# assert_eq!(AnyName::One,                     oxixml_xml::de::from_str(r#"<one field1="...">...</one>"#).expect("should be Ok"));
# assert_eq!(AnyName::Two(Two { field2: () }), oxixml_xml::de::from_str(r#"<two><field2>...</field2></two>"#).expect("should be Ok"));
# assert_eq!(AnyName::Text,                    oxixml_xml::de::from_str(r#"text <![CDATA[ cdata ]]>"#).expect("should be Ok"));
```
```
# use serde::Deserialize;
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnyName {
  One,
  // the <two> and textual content will be mapped to this
  #[serde(other)]
  Other,
}
# assert_eq!(AnyName::One,   oxixml_xml::de::from_str(r#"<one field1="...">...</one>"#).expect("should be Ok"));
# assert_eq!(AnyName::Other, oxixml_xml::de::from_str(r#"<two><field2>...</field2></two>"#).expect("should be Ok"));
# assert_eq!(AnyName::Other, oxixml_xml::de::from_str(r#"text <![CDATA[ cdata ]]>"#).expect("should be Ok"));
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: You should have variants for all possible tag names in your enum
or have an `#[serde(other)]` variant. Without either, an unrecognized tag
name does not panic: [`serde`] itself rejects it with
`serde::de::Error::unknown_variant`, surfaced through this crate as
[`DeError::Custom`].
</div>
</td>
</tr>
<!-- 9 ===================================================================================== -->
<tr>
<td>

`<xs:choice>` embedded in the other element, and at the same time you want
to get access to other attributes that can appear in the same container
(`<any-tag>`). Also this case can be described, as if you want to choose
Rust enum variant based on a tag name:

```xml
<any-tag field="...">
  <one>...</one>
</any-tag>
```
```xml
<any-tag field="...">
  <two>...</two>
</any-tag>
```
```xml
<any-tag field="...">
  Text <![CDATA[or (mixed)
  CDATA]]> content
</any-tag>
```
</td>
<td>

A structure with a field which type is an `enum`.

If you need to get a textual content, mark a variant with `#[serde(rename = "$text")]`.

Names of the enum, struct, and struct field with `Choice` type does not matter:

```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,

  /// Use unit variant, if you do not care of a content.
  /// You can use tuple variant if you want to parse
  /// textual content as an xs:list.
  /// Struct variants are will pass a string to the
  /// struct enum variant visitor, which typically
  /// returns Err(Custom)
  #[serde(rename = "$text")]
  Text(String),
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@field")]
  field: T,

  #[serde(rename = "$value")]
  any_name: Choice,
}
# assert_eq!(
#   AnyName { field: (), any_name: Choice::One },
#   oxixml_xml::de::from_str(r#"<any-tag field="..."><one>...</one></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { field: (), any_name: Choice::Two },
#   oxixml_xml::de::from_str(r#"<any-tag field="..."><two>...</two></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { field: (), any_name: Choice::Text("text  cdata ".into()) },
#   oxixml_xml::de::from_str(r#"<any-tag field="...">text <![CDATA[ cdata ]]></any-tag>"#).expect("should be Ok"),
# );
```
</td>
</tr>
<!-- 10 ==================================================================================== -->
<tr>
<td>

`<xs:choice>` embedded in the other element, and at the same time you want
to get access to other elements that can appear in the same container
(`<any-tag>`). Also this case can be described, as if you want to choose
Rust enum variant based on a tag name:

```xml
<any-tag>
  <field>...</field>
  <one>...</one>
</any-tag>
```
```xml
<any-tag>
  <two>...</two>
  <field>...</field>
</any-tag>
```
</td>
<td>

A structure with a field which type is an `enum`.

Names of the enum, struct, and struct field with `Choice` type does not matter:

```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  field: T,

  #[serde(rename = "$value")]
  any_name: Choice,
}
# assert_eq!(
#   AnyName { field: (), any_name: Choice::One },
#   oxixml_xml::de::from_str(r#"<any-tag><field>...</field><one>...</one></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { field: (), any_name: Choice::Two },
#   oxixml_xml::de::from_str(r#"<any-tag><two>...</two><field>...</field></any-tag>"#).expect("should be Ok"),
# );
```

<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: if your `Choice` enum would contain an `#[serde(other)]`
variant, element `<field>` will be mapped to the `field` and not to the enum
variant.
</div>

</td>
</tr>
<!-- 11 ==================================================================================== -->
<tr>
<td>

`<xs:choice>` encapsulated in other element with a fixed name:

```xml
<any-tag field="...">
  <choice>
    <one>...</one>
  </choice>
</any-tag>
```
```xml
<any-tag field="...">
  <choice>
    <two>...</two>
  </choice>
</any-tag>
```
</td>
<td>

A structure with a field of an intermediate type with one field of `enum` type.
Actually, this example is not necessary, because you can construct it by yourself
using the composition rules that were described above. However the XML construction
described here is very common, so it is shown explicitly.

Names of the enum and struct does not matter:

```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct Holder {
  #[serde(rename = "$value")]
  any_name: Choice,
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@field")]
  field: T,

  choice: Holder,
}
# assert_eq!(
#   AnyName { field: (), choice: Holder { any_name: Choice::One } },
#   oxixml_xml::de::from_str(r#"<any-tag field="..."><choice><one>...</one></choice></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { field: (), choice: Holder { any_name: Choice::Two } },
#   oxixml_xml::de::from_str(r#"<any-tag field="..."><choice><two>...</two></choice></any-tag>"#).expect("should be Ok"),
# );
```
</td>
</tr>
<!-- 12 ==================================================================================== -->
<tr>
<td>

`<xs:choice>` encapsulated in other element with a fixed name:

```xml
<any-tag>
  <field>...</field>
  <choice>
    <one>...</one>
  </choice>
</any-tag>
```
```xml
<any-tag>
  <choice>
    <two>...</two>
  </choice>
  <field>...</field>
</any-tag>
```
</td>
<td>

A structure with a field of an intermediate type with one field of `enum` type.
Actually, this example is not necessary, because you can construct it by yourself
using the composition rules that were described above. However the XML construction
described here is very common, so it is shown explicitly.

Names of the enum and struct does not matter:

```
# use serde::Deserialize;
# type T = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct Holder {
  #[serde(rename = "$value")]
  any_name: Choice,
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  field: T,

  choice: Holder,
}
# assert_eq!(
#   AnyName { field: (), choice: Holder { any_name: Choice::One } },
#   oxixml_xml::de::from_str(r#"<any-tag><field>...</field><choice><one>...</one></choice></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { field: (), choice: Holder { any_name: Choice::Two } },
#   oxixml_xml::de::from_str(r#"<any-tag><choice><two>...</two></choice><field>...</field></any-tag>"#).expect("should be Ok"),
# );
```
</td>
</tr>
<!-- ======================================================================================== -->
<tr><th colspan="2">

## Sequences (`xs:all` and `xs:sequence` XML Schema types)

</th></tr>
<tr><th>To parse all these XML's...</th><th>...use these Rust type(s)</th></tr>
<!-- 13 ==================================================================================== -->
<tr>
<td>
A sequence inside of a tag without a dedicated name:

```xml
<any-tag/>
```
```xml
<any-tag>
  <item/>
</any-tag>
```
```xml
<any-tag>
  <item/>
  <item/>
  <item/>
</any-tag>
```
</td>
<td>

A structure with a field which is a sequence type, for example, [`Vec`].
Because XML syntax does not distinguish between empty sequences and missed
elements, we should indicate that on the Rust side, because serde will require
that field `item` exists. You can do that in two possible ways:

Use the `#[serde(default)]` attribute for a [field] or the entire [struct]:
```
# use serde::Deserialize;
# type Item = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(default)]
  item: Vec<Item>,
}
# assert_eq!(
#   AnyName { item: vec![] },
#   oxixml_xml::de::from_str(r#"<any-tag/>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { item: vec![()] },
#   oxixml_xml::de::from_str(r#"<any-tag><item/></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { item: vec![(), (), ()] },
#   oxixml_xml::de::from_str(r#"<any-tag><item/><item/><item/></any-tag>"#).expect("should be Ok"),
# );
```

Use the [`Option`]. In that case inner array will always contains at least one
element after deserialization:
```ignore
# use serde::Deserialize;
# type Item = ();
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  item: Option<Vec<Item>>,
}
# assert_eq!(
#   AnyName { item: None },
#   oxixml_xml::de::from_str(r#"<any-tag/>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { item: Some(vec![()]) },
#   oxixml_xml::de::from_str(r#"<any-tag><item/></any-tag>"#).expect("should be Ok"),
# );
# assert_eq!(
#   AnyName { item: Some(vec![(), (), ()]) },
#   oxixml_xml::de::from_str(r#"<any-tag><item/><item/><item/></any-tag>"#).expect("should be Ok"),
# );
```

See also [Frequently Used Patterns](#element-lists).

[field]: https://serde.rs/field-attrs.html#default
[struct]: https://serde.rs/container-attrs.html#default
</td>
</tr>
<!-- 14 ==================================================================================== -->
<tr>
<td>
A sequence with a strict order, probably with mixed content
(text / CDATA and tags):

```xml
<one>...</one>
text
<![CDATA[cdata]]>
<two>...</two>
<one>...</one>
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: this is just an example for showing mapping. XML does not allow
multiple root tags -- you should wrap the sequence into a tag.
</div>
</td>
<td>

All elements mapped to the heterogeneous sequential type: tuple or named tuple.
Each element of the tuple should be able to be deserialized from the nested
element content (`...`), except the enum types which would be deserialized
from the full element (`<one>...</one>`), so they could use the element name
to choose the right variant:

```
# use serde::Deserialize;
# type One = ();
# type Two = ();
# /*
type One = ...;
type Two = ...;
# */
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName(One, String, Two, One);
# assert_eq!(
#   AnyName((), "text cdata".into(), (), ()),
#   oxixml_xml::de::from_str(r#"<one>...</one>text <![CDATA[cdata]]><two>...</two><one>...</one>"#).expect("should be Ok"),
# );
```
```
# use serde::Deserialize;
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
}
# type Two = ();
# /*
type Two = ...;
# */
type AnyName = (Choice, String, Two, Choice);
# assert_eq!(
#   (Choice::One, "text cdata".to_string(), (), Choice::One),
#   oxixml_xml::de::from_str(r#"<one>...</one>text <![CDATA[cdata]]><two>...</two><one>...</one>"#).expect("should be Ok"),
# );
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: consequent text and CDATA nodes are merged into the one text node,
so you cannot have two adjacent string types in your sequence.

NOTE: In the case that the list might contain tags that are overlapped with
tags that do not correspond to the list you should add the feature [`overlapped-lists`].
</div>
</td>
</tr>
<!-- 15 ==================================================================================== -->
<tr>
<td>
A sequence with a non-strict order, probably with a mixed content
(text / CDATA and tags).

```xml
<one>...</one>
text
<![CDATA[cdata]]>
<two>...</two>
<one>...</one>
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: this is just an example for showing mapping. XML does not allow
multiple root tags -- you should wrap the sequence into a tag.
</div>
</td>
<td>
A homogeneous sequence of elements with a fixed or dynamic size:

```
# use serde::Deserialize;
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
  #[serde(other)]
  Other,
}
type AnyName = [Choice; 4];
# assert_eq!(
#   [Choice::One, Choice::Other, Choice::Two, Choice::One],
#   oxixml_xml::de::from_str::<AnyName>(r#"<one>...</one>text <![CDATA[cdata]]><two>...</two><one>...</one>"#).expect("should be Ok"),
# );
```
```
# use serde::Deserialize;
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
  #[serde(rename = "$text")]
  Other(String),
}
type AnyName = Vec<Choice>;
# assert_eq!(
#   vec![
#     Choice::One,
#     Choice::Other("text cdata".into()),
#     Choice::Two,
#     Choice::One,
#   ],
#   oxixml_xml::de::from_str::<AnyName>(r#"<one>...</one>text <![CDATA[cdata]]><two>...</two><one>...</one>"#).expect("should be Ok"),
# );
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: consequent text and CDATA nodes are merged into the one text node,
so you cannot have two adjacent string types in your sequence.
</div>
</td>
</tr>
<!-- 16 ==================================================================================== -->
<tr>
<td>
A sequence with a strict order, probably with a mixed content,
(text and tags) inside of the other element:

```xml
<any-tag attribute="...">
  <one>...</one>
  text
  <![CDATA[cdata]]>
  <two>...</two>
  <one>...</one>
</any-tag>
```
</td>
<td>

A structure where all child elements mapped to the one field which have
a heterogeneous sequential type: tuple or named tuple. Each element of the
tuple should be able to be deserialized from the full element (`<one>...</one>`).

You MUST specify `#[serde(rename = "$value")]` on that field:

```
# use serde::Deserialize;
# type One = ();
# type Two = ();
# /*
type One = ...;
type Two = ...;
# */

# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@attribute")]
# attribute: (),
# /*
  attribute: ...,
# */
  // Does not (yet?) supported by the serde
  // https://github.com/serde-rs/serde/issues/1905
  // #[serde(flatten)]
  #[serde(rename = "$value")]
  any_name: (One, String, Two, One),
}
# assert_eq!(
#   AnyName { attribute: (), any_name: ((), "text cdata".into(), (), ()) },
#   oxixml_xml::de::from_str("\
#     <any-tag attribute='...'>\
#       <one>...</one>\
#       text \
#       <![CDATA[cdata]]>\
#       <two>...</two>\
#       <one>...</one>\
#     </any-tag>"
#   ).expect("should be Ok"),
# );
```
```
# use serde::Deserialize;
# type One = ();
# type Two = ();
# /*
type One = ...;
type Two = ...;
# */

# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct NamedTuple(One, String, Two, One);

# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@attribute")]
# attribute: (),
# /*
  attribute: ...,
# */
  // Does not (yet?) supported by the serde
  // https://github.com/serde-rs/serde/issues/1905
  // #[serde(flatten)]
  #[serde(rename = "$value")]
  any_name: NamedTuple,
}
# assert_eq!(
#   AnyName { attribute: (), any_name: NamedTuple((), "text cdata".into(), (), ()) },
#   oxixml_xml::de::from_str("\
#     <any-tag attribute='...'>\
#       <one>...</one>\
#       text \
#       <![CDATA[cdata]]>\
#       <two>...</two>\
#       <one>...</one>\
#     </any-tag>"
#   ).expect("should be Ok"),
# );
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: consequent text and CDATA nodes are merged into the one text node,
so you cannot have two adjacent string types in your sequence.
</div>
</td>
</tr>
<!-- 17 ==================================================================================== -->
<tr>
<td>
A sequence with a non-strict order, probably with a mixed content
(text / CDATA and tags) inside of the other element:

```xml
<any-tag>
  <one>...</one>
  text
  <![CDATA[cdata]]>
  <two>...</two>
  <one>...</one>
</any-tag>
```
</td>
<td>

A structure where all child elements mapped to the one field which have
a homogeneous sequential type: array-like container. A container type `T`
should be able to be deserialized from the nested element content (`...`),
except if it is an enum type which would be deserialized from the full
element (`<one>...</one>`).

You MUST specify `#[serde(rename = "$value")]` on that field:

```
# use serde::Deserialize;
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
  #[serde(rename = "$text")]
  Other(String),
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@attribute")]
# attribute: (),
# /*
  attribute: ...,
# */
  // Does not (yet?) supported by the serde
  // https://github.com/serde-rs/serde/issues/1905
  // #[serde(flatten)]
  #[serde(rename = "$value")]
  any_name: [Choice; 4],
}
# assert_eq!(
#   AnyName { attribute: (), any_name: [
#     Choice::One,
#     Choice::Other("text cdata".into()),
#     Choice::Two,
#     Choice::One,
#   ] },
#   oxixml_xml::de::from_str("\
#     <any-tag attribute='...'>\
#       <one>...</one>\
#       text \
#       <![CDATA[cdata]]>\
#       <two>...</two>\
#       <one>...</one>\
#     </any-tag>"
#   ).expect("should be Ok"),
# );
```
```
# use serde::Deserialize;
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
  One,
  Two,
  #[serde(rename = "$text")]
  Other(String),
}
# #[derive(Debug, PartialEq)]
#[derive(Deserialize)]
struct AnyName {
  #[serde(rename = "@attribute")]
# attribute: (),
# /*
  attribute: ...,
# */
  // Does not (yet?) supported by the serde
  // https://github.com/serde-rs/serde/issues/1905
  // #[serde(flatten)]
  #[serde(rename = "$value")]
  any_name: Vec<Choice>,
}
# assert_eq!(
#   AnyName { attribute: (), any_name: vec![
#     Choice::One,
#     Choice::Other("text cdata".into()),
#     Choice::Two,
#     Choice::One,
#   ] },
#   oxixml_xml::de::from_str("\
#     <any-tag attribute='...'>\
#       <one>...</one>\
#       text \
#       <![CDATA[cdata]]>\
#       <two>...</two>\
#       <one>...</one>\
#     </any-tag>"
#   ).expect("should be Ok"),
# );
```
<div style="background:rgba(120,145,255,0.45);padding:0.75em;">

NOTE: consequent text and CDATA nodes are merged into the one text node,
so you cannot have two adjacent string types in your sequence.
</div>
</td>
</tr>
</tbody>
</table>


Mapping of `xsi:nil`
====================

quick-xml supports handling of [`xsi:nil`] special attribute. When field of optional
type is mapped to the XML element which have `xsi:nil="true"` set, or if that attribute
is placed on parent XML element, the deserializer will call [`Visitor::visit_none`](serde::de::Visitor::visit_none)
and skip XML element corresponding to a field.

Examples:

```
# use serde::Deserialize;
#[derive(Deserialize, Debug, PartialEq)]
struct TypeWithOptionalField {
  element: Option<String>,
}

assert_eq!(
  TypeWithOptionalField {
    element: None,
  },
  oxixml_xml::de::from_str("
    <any-tag xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
      <element xsi:nil='true'>Content is skiped because of xsi:nil='true'</element>
    </any-tag>
  ").expect("should be Ok"),
);
```

You can capture attributes from the optional type, because ` xsi:nil="true"` elements can have
attributes:
```
# use serde::Deserialize;
#[derive(Deserialize, Debug, PartialEq)]
struct TypeWithOptionalField {
  #[serde(rename = "@attribute")]
  attribute: usize,

  element: Option<String>,
  non_optional: String,
}

assert_eq!(
  TypeWithOptionalField {
    attribute: 42,
    element: None,
    non_optional: "Note, that non-optional fields will be deserialized as usual".to_string(),
  },
  oxixml_xml::de::from_str("
    <any-tag attribute='42' xsi:nil='true' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
      <element>Content is skiped because of xsi:nil='true'</element>
      <non_optional>Note, that non-optional fields will be deserialized as usual</non_optional>
    </any-tag>
  ").expect("should be Ok"),
);
```

Generate Rust types from XML
============================

To speed up the creation of Rust types that represent a given XML file you can
use the [xml_schema_generator](https://github.com/Thomblin/xml_schema_generator).
It provides a standalone binary and a Rust library that parses one or more XML files
and generates a collection of structs that are compatible with oxixml_xml::de.



Composition Rules
=================

The XML format is very different from other formats supported by `serde`.
One such difference it is how data in the serialized form is related to
the Rust type. Usually each byte in the data can be associated only with
one field in the data structure. However, XML is an exception.

For example, took this XML:

```xml
<any>
  <key attr="value"/>
</any>
```

and try to deserialize it to the struct `AnyName`:

```no_run
# use serde::Deserialize;
#[derive(Deserialize)]
struct AnyName { // AnyName calls `deserialize_struct` on `<any><key attr="value"/></any>`
                 //                         Used data:          ^^^^^^^^^^^^^^^^^^^
  key: Inner,    // Inner   calls `deserialize_struct` on `<key attr="value"/>`
                 //                         Used data:          ^^^^^^^^^^^^
}
#[derive(Deserialize)]
struct Inner {
  #[serde(rename = "@attr")]
  attr: String,  // String  calls `deserialize_string` on `value`
                 //                         Used data:     ^^^^^
}
```

Comments shows what methods of a [`Deserializer`] called by each struct
`deserialize` method and which input their seen. **Used data** shows, what
content is actually used for deserializing. As you see, name of the inner
`<key>` tag used both as a map key / outer struct field name and as part
of the inner struct (although _value_ of the tag, i.e. `key` is not used
by it).



Enum Representations
====================

`quick-xml` represents enums differently in normal fields, `$text` fields and
`$value` fields. A normal representation is compatible with serde's adjacent
and internal tags feature -- tag for adjacently and internally tagged enums
are serialized using [`Serializer::serialize_unit_variant`] and deserialized
using [`Deserializer::deserialize_enum`].

Use those simple rules to remember, how enum would be represented in XML:
- In `$value` field the representation is always the same as top-level representation;
- In `$text` field the representation is always the same as in normal field,
  but surrounding tags with field name are removed;
- In normal field the representation is always contains a tag with a field name.

Normal enum variant
-------------------

To model an `xs:choice` XML construct use `$value` field.
To model a top-level `xs:choice` just use the enum type.

|Kind   |Top-level and in `$value` field          |In normal field      |In `$text` field     |
|-------|-----------------------------------------|---------------------|---------------------|
|Unit   |`<Unit/>`                                |`<field>Unit</field>`|`Unit`               |
|Newtype|`<Newtype>42</Newtype>`                  |Err(Custom) [^0]     |Err(Custom) [^0]     |
|Tuple  |`<Tuple>42</Tuple><Tuple>answer</Tuple>` |Err(Custom) [^0]     |Err(Custom) [^0]     |
|Struct |`<Struct><q>42</q><a>answer</a></Struct>`|Err(Custom) [^0]     |Err(Custom) [^0]     |

`$text` enum variant
--------------------

|Kind   |Top-level and in `$value` field          |In normal field      |In `$text` field     |
|-------|-----------------------------------------|---------------------|---------------------|
|Unit   |_(empty)_                                |`<field/>`           |_(empty)_            |
|Newtype|`42`                                     |Err(Custom) [^0] [^1]|Err(Custom) [^0] [^2]|
|Tuple  |`42 answer`                              |Err(Custom) [^0] [^3]|Err(Custom) [^0] [^4]|
|Struct |Err(Custom) [^0]                         |Err(Custom) [^0]     |Err(Custom) [^0]     |

[^0]: Error is returned by the deserialized type. In case of derived implementation a `Custom`
      error will be returned, but custom deserialize implementation can successfully deserialize
      value from a string which will be passed to it.

[^1]: If this serialize as `<field>42</field>` then it will be ambiguity during deserialization,
      because it clash with `Unit` representation in normal field.

[^2]: If this serialize as `42` then it will be ambiguity during deserialization,
      because it clash with `Unit` representation in `$text` field.

[^3]: If this serialize as `<field>42 answer</field>` then it will be ambiguity during deserialization,
      because it clash with `Unit` representation in normal field.

[^4]: If this serialize as `42 answer` then it will be ambiguity during deserialization,
      because it clash with `Unit` representation in `$text` field.



`$text` and `$value` special names
==================================

quick-xml supports two special names for fields -- `$text` and `$value`.
Although they may seem the same, there is a distinction. Two different
names is required mostly for serialization, because quick-xml should know
how you want to serialize certain constructs, which could be represented
through XML in multiple different ways.

The only difference is in how complex types and sequences are serialized.
If you doubt which one you should select, begin with [`$value`](#value).

If you have both `$text` and `$value` in you struct, then text events will be
mapped to the `$text` field:

```
# use serde::Deserialize;
# use oxixml_xml::de::from_str;
#[derive(Deserialize, PartialEq, Debug)]
struct TextAndValue {
    #[serde(rename = "$text")]
    text: Option<String>,

    #[serde(rename = "$value")]
    value: Option<String>,
}

let object: TextAndValue = from_str("<AnyName>text <![CDATA[and CDATA]]></AnyName>").expect("should be Ok");
assert_eq!(object, TextAndValue {
    text: Some("text and CDATA".to_string()),
    value: None,
});
```
