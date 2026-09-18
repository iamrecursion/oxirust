//! Port of the quick-xml 0.41 `tests/serde-se.rs` suite for the
//! [`oxixml_xml::se`] serializer.
//!
//! Shared fixture types live here; the actual assertions live in the
//! `without_root` and `with_root` submodules.

use oxixml_xml::de::from_str;
use oxixml_xml::se::{SeError, Serializer};
use oxixml_xml::utils::Bytes;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Unit;

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Newtype(bool);

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Tuple(f32, &'static str);

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Struct {
    float: f64,
    string: &'static str,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct NestedStruct {
    nested: Nested,
    string: &'static str,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct FlattenStruct {
    #[serde(flatten)]
    nested: Nested,
    string: &'static str,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Nested {
    float: f64,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Empty {}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct EmptyWithAttribute {
    #[serde(rename = "@attr")]
    attr: f64,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Text {
    #[serde(rename = "$text")]
    float: f64,
    string: &'static str,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
enum ExternallyTagged {
    Unit,
    Newtype(bool),
    Tuple(f64, &'static str),
    Struct {
        float: f64,
        string: &'static str,
    },
    Holder {
        nested: Nested,
        string: &'static str,
    },
    Flatten {
        #[serde(flatten)]
        nested: Nested,
        string: &'static str,
    },
    /// `float` field serialized as textual content instead of a tag
    Text {
        #[serde(rename = "$text")]
        float: f64,
        string: &'static str,
    },
    Empty {},
    EmptyWithAttribute {
        #[serde(rename = "@attr")]
        attr: f64,
    },
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "tag")]
enum InternallyTagged {
    Unit,
    /// Primitives (such as `bool`) are not supported by the serde in the internally tagged mode
    Newtype(Nested),
    // Tuple(f64, &'static str),// Tuples are not supported in the internally tagged mode
    Struct {
        float: f64,
        string: &'static str,
    },
    Holder {
        nested: Nested,
        string: &'static str,
    },
    Flatten {
        #[serde(flatten)]
        nested: Nested,
        string: &'static str,
    },
    /// `float` field serialized as textual content instead of a tag
    Text {
        #[serde(rename = "$text")]
        float: f64,
        string: &'static str,
    },
    Empty {},
    EmptyWithAttribute {
        #[serde(rename = "@attr")]
        attr: f64,
    },
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "tag", content = "content")]
enum AdjacentlyTagged {
    Unit,
    Newtype(bool),
    Tuple(f64, &'static str),
    Struct {
        float: f64,
        string: &'static str,
    },
    Holder {
        nested: Nested,
        string: &'static str,
    },
    Flatten {
        #[serde(flatten)]
        nested: Nested,
        string: &'static str,
    },
    /// `float` field serialized as textual content instead of a tag
    Text {
        #[serde(rename = "$text")]
        float: f64,
        string: &'static str,
    },
    Empty {},
    EmptyWithAttribute {
        #[serde(rename = "@attr")]
        attr: f64,
    },
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
enum Untagged {
    Unit,
    Newtype(bool),
    Tuple(f64, &'static str),
    Struct {
        float: f64,
        string: &'static str,
    },
    Holder {
        nested: Nested,
        string: &'static str,
    },
    Flatten {
        #[serde(flatten)]
        nested: Nested,
        string: &'static str,
    },
    /// `float` field serialized as textual content instead of a tag
    Text {
        #[serde(rename = "$text")]
        float: f64,
        string: &'static str,
    },
    Empty {},
    EmptyWithAttribute {
        #[serde(rename = "@attr")]
        attr: f64,
    },
}

#[path = "serde-se/with_root.rs"]
mod with_root;

#[path = "serde-se/without_root.rs"]
mod without_root;
