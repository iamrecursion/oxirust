//! Tests of deserialization of XML documents into various sequential types

use oxixml_xml::DeError;
use serde::Deserialize;

mod serde_helpers;
use serde_helpers::from_str;

/// Check that top-level sequences can be deserialized from the multi-root XML documents
#[path = "serde-de-seq/top_level.rs"]
mod top_level;

/// Tests where each sequence item have an identical name in an XML.
/// That explicitly means that `enum`s as list elements are not supported
/// in that case, because enum requires different tags.
///
/// (by `enums` we mean [externally tagged enums] is serde terminology)
///
/// [externally tagged enums]: https://serde.rs/enum-representations.html#externally-tagged
#[path = "serde-de-seq/fixed_name.rs"]
mod fixed_name;

/// Check that sequences inside element can be deserialized.
/// In terms of serde this is a sequence flatten into the struct:
///
/// ```ignore
/// struct Root {
///   #[serde(flatten)]
///   items: Vec<T>,
/// }
/// ```
/// except that fact that this is not supported nowadays
/// (https://github.com/serde-rs/serde/issues/1905)
///
/// Because this is very frequently used pattern in the XML, quick-xml
/// have a workaround for this. If a field will have a special name `$value`
/// then any `xs:element`s in the `xs:sequence` / `xs:all`, except that
/// which name matches the struct name, will be associated with this field:
///
/// ```ignore
/// struct Root {
///   field: U,
///   #[serde(rename = "$value")]
///   items: Vec<Enum>,
/// }
/// ```
/// In this example `<field>` tag will be associated with a `field` field,
/// but all other tags will be associated with an `items` field. Disadvantages
/// of this approach that you can have only one field, but usually you don't
/// want more
#[path = "serde-de-seq/variable_name.rs"]
mod variable_name;
