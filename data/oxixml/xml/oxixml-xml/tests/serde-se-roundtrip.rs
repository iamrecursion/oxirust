//! Round-trip tests: `to_string` -> `from_str` must reproduce the original value.
//!
//! The corpus below covers the shapes that the COOLJAPAN consumers of this crate
//! actually serialize:
//!
//! 1. *configuration documents* — plain structs of scalars nested a few levels
//!    deep (`oxixml-relaxng`, `oxixml-schematron` grammars);
//! 2. *attribute-carrying elements* — `@name`-style fields mixed with child
//!    elements (`oxixml-rdfxml`, `oxixml-trix` node descriptions);
//! 3. *mixed content* — an element that has both attributes and a `$text` body
//!    (`oxixml-dsig` / `oxixml-c14n` signature values);
//! 4. *repeated children* — `Vec<T>` of structs and of scalars under a fixed tag
//!    (`oxixml-sparql-results` bindings);
//! 5. *choice content* — externally tagged enums serialized through `$value`
//!    (`oxixml-xslt` instruction bodies).
//!
//! Each test asserts both the exact serialized form (so that a regression in the
//! writer is caught immediately) and that deserializing it yields the input.

use oxixml_xml::de::from_str;
use oxixml_xml::se::{Serializer, to_string, to_string_with_root};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Serializes `value` with the root tag deduced from the type, checks the
/// produced XML against `expected` and then deserializes it back.
#[track_caller]
fn round_trip<T>(value: &T, expected: &str)
where
    T: Debug + PartialEq + Serialize + DeserializeOwned,
{
    let xml = to_string(value).expect("serialization should succeed");
    assert_eq!(xml, expected, "serialized form");

    let back: T = from_str(&xml).expect("deserialization should succeed");
    assert_eq!(&back, value, "round trip");
}

/// The same as [`round_trip`], but with an explicitly specified root tag.
#[track_caller]
fn round_trip_with_root<T>(root: &str, value: &T, expected: &str)
where
    T: Debug + PartialEq + Serialize + DeserializeOwned,
{
    let xml = to_string_with_root(root, value).expect("serialization should succeed");
    assert_eq!(xml, expected, "serialized form");

    let back: T = from_str(&xml).expect("deserialization should succeed");
    assert_eq!(&back, value, "round trip");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Configuration documents: nested structs of scalars
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Limits {
    depth: u32,
    ratio: f64,
    strict: bool,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Config {
    name: String,
    limits: Limits,
}

#[test]
fn nested_struct_of_scalars() {
    round_trip(
        &Config {
            name: "default".to_string(),
            limits: Limits {
                depth: 64,
                ratio: 0.5,
                strict: true,
            },
        },
        "<Config>\
            <name>default</name>\
            <limits>\
                <depth>64</depth>\
                <ratio>0.5</ratio>\
                <strict>true</strict>\
            </limits>\
        </Config>",
    );
}

#[test]
fn nested_struct_with_indent() -> Result<(), oxixml_xml::se::SeError> {
    let data = Config {
        name: "default".to_string(),
        limits: Limits {
            depth: 64,
            ratio: 0.5,
            strict: true,
        },
    };

    let mut buffer = String::new();
    let mut ser = Serializer::with_root(&mut buffer, Some("config"))?;
    ser.indent(' ', 2);
    data.serialize(ser)?;

    assert_eq!(
        buffer,
        "<config>\n  \
           <name>default</name>\n  \
           <limits>\n    \
             <depth>64</depth>\n    \
             <ratio>0.5</ratio>\n    \
             <strict>true</strict>\n  \
           </limits>\n\
         </config>"
    );

    let back: Config = from_str(&buffer).expect("deserialization should succeed");
    assert_eq!(back, data);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2. Attribute-carrying elements
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Resource {
    #[serde(rename = "@about")]
    about: String,
    #[serde(rename = "@lang")]
    lang: String,
    label: String,
}

#[test]
fn attributes_and_child_element() {
    round_trip(
        &Resource {
            about: "http://example.com/#it".to_string(),
            lang: "en".to_string(),
            label: "Example".to_string(),
        },
        "<Resource about=\"http://example.com/#it\" lang=\"en\">\
            <label>Example</label>\
        </Resource>",
    );
}

#[test]
fn attribute_values_are_escaped() {
    round_trip(
        &Resource {
            about: "a<b>c&d\"e".to_string(),
            lang: "x".to_string(),
            label: "x<y>z&w\"v".to_string(),
        },
        // `"` must be escaped inside a double-quoted attribute, but may stay
        // literal in text content.
        "<Resource about=\"a&lt;b&gt;c&amp;d&quot;e\" lang=\"x\">\
            <label>x&lt;y&gt;z&amp;w\"v</label>\
        </Resource>",
    );
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Attributes {
    #[serde(rename = "@id")]
    id: u32,
    #[serde(rename = "@kind")]
    kind: Kind,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Kind {
    Literal,
    Uri,
}

#[test]
fn attribute_only_struct_is_self_closed() {
    round_trip(
        &Attributes {
            id: 7,
            kind: Kind::Uri,
        },
        "<Attributes id=\"7\" kind=\"Uri\"/>",
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 3. Mixed content: attributes + `$text`
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SignatureValue {
    #[serde(rename = "@Id")]
    id: String,
    #[serde(rename = "$text")]
    value: String,
}

#[test]
fn attribute_plus_text() {
    round_trip_with_root(
        "SignatureValue",
        &SignatureValue {
            id: "sig-1".to_string(),
            value: "MC0CFQ==".to_string(),
        },
        "<SignatureValue Id=\"sig-1\">MC0CFQ==</SignatureValue>",
    );
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct TextAndElements {
    #[serde(rename = "@name")]
    name: String,
    child: u32,
    #[serde(rename = "$text")]
    text: String,
}

#[test]
fn attribute_text_and_nested_element() {
    round_trip(
        &TextAndElements {
            name: "n".to_string(),
            child: 1,
            text: "trailing".to_string(),
        },
        "<TextAndElements name=\"n\"><child>1</child>trailing</TextAndElements>",
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 4. Repeated children
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Binding {
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "$text")]
    value: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Solution {
    // An empty `Vec` produces no elements at all, so deserializing it back
    // requires the field to be defaultable.
    #[serde(default)]
    binding: Vec<Binding>,
}

#[test]
fn sequence_of_structs_under_fixed_tag() {
    round_trip(
        &Solution {
            binding: vec![
                Binding {
                    name: "s".to_string(),
                    value: "http://example.com/s".to_string(),
                },
                Binding {
                    name: "o".to_string(),
                    value: "42".to_string(),
                },
            ],
        },
        "<Solution>\
            <binding name=\"s\">http://example.com/s</binding>\
            <binding name=\"o\">42</binding>\
        </Solution>",
    );
}

#[test]
fn empty_sequence_produces_self_closed_root() {
    round_trip(&Solution { binding: vec![] }, "<Solution/>");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Numbers {
    item: Vec<u32>,
}

#[test]
fn sequence_of_scalars_under_fixed_tag() {
    round_trip(
        &Numbers {
            item: vec![1, 2, 3],
        },
        "<Numbers><item>1</item><item>2</item><item>3</item></Numbers>",
    );
}

#[test]
fn map_of_scalars() {
    let mut map = BTreeMap::new();
    map.insert("alpha".to_string(), 1u32);
    map.insert("beta".to_string(), 2u32);

    round_trip_with_root("map", &map, "<map><alpha>1</alpha><beta>2</beta></map>");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 5. Choice content: enums in `$value`
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Instruction {
    ValueOf {
        #[serde(rename = "@select")]
        select: String,
    },
    Text(String),
    Fallback,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Template {
    #[serde(rename = "@match")]
    pattern: String,
    #[serde(rename = "$value")]
    body: Vec<Instruction>,
}

#[test]
fn externally_tagged_enum_sequence_in_value() {
    round_trip(
        &Template {
            pattern: "/root".to_string(),
            body: vec![
                Instruction::ValueOf {
                    select: "@id".to_string(),
                },
                Instruction::Text("hello".to_string()),
                Instruction::Fallback,
            ],
        },
        "<Template match=\"/root\">\
            <ValueOf select=\"@id\"/>\
            <Text>hello</Text>\
            <Fallback/>\
        </Template>",
    );
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Choice {
    #[serde(rename = "$value")]
    value: Instruction,
}

#[test]
fn externally_tagged_enum_single_in_value() {
    round_trip(
        &Choice {
            value: Instruction::Text("body".to_string()),
        },
        "<Choice><Text>body</Text></Choice>",
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Miscellaneous shapes
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Optional {
    #[serde(skip_serializing_if = "Option::is_none")]
    present: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    absent: Option<u32>,
}

#[test]
fn skipped_none_is_absent_after_round_trip() {
    round_trip(
        &Optional {
            present: Some(1),
            absent: None,
        },
        "<Optional><present>1</present></Optional>",
    );
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Newtype(u64);

#[test]
fn newtype_struct() {
    round_trip_with_root("Newtype", &Newtype(42), "<Newtype>42</Newtype>");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Unit;

#[test]
fn unit_struct() {
    round_trip_with_root("Unit", &Unit, "<Unit/>");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Nested {
    inner: Limits,
}

#[test]
fn deeply_nested_struct() {
    round_trip(
        &Nested {
            inner: Limits {
                depth: 1,
                ratio: 2.5,
                strict: false,
            },
        },
        "<Nested>\
            <inner>\
                <depth>1</depth>\
                <ratio>2.5</ratio>\
                <strict>false</strict>\
            </inner>\
        </Nested>",
    );
}

#[test]
fn text_special_characters_survive_round_trip() {
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Body {
        #[serde(rename = "$text")]
        text: String,
    }

    round_trip_with_root(
        "body",
        &Body {
            text: "1 < 2 && 3 > 2".to_string(),
        },
        "<body>1 &lt; 2 &amp;&amp; 3 &gt; 2</body>",
    );
}

#[test]
fn serializer_expand_empty_elements_round_trips() -> Result<(), oxixml_xml::se::SeError> {
    let data = Optional {
        present: None,
        absent: None,
    };

    let mut buffer = String::new();
    let mut ser = Serializer::with_root(&mut buffer, Some("Optional"))?;
    ser.expand_empty_elements(true);
    data.serialize(ser)?;

    assert_eq!(buffer, "<Optional></Optional>");

    let back: Optional = from_str(&buffer).expect("deserialization should succeed");
    assert_eq!(back, data);
    Ok(())
}
