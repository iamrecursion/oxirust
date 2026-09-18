//! The `quick_xml::serde_helpers` surface, exercised through the compat alias.

use oxixml_quickxml_compat as quick_xml;
use quick_xml::de::from_str;
use quick_xml::impl_deserialize_for_internally_tagged_enum;
use quick_xml::se::to_string;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
enum SomeEnum {
    EnumValue,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
#[serde(rename = "some-container")]
struct SomeContainer {
    #[serde(with = "quick_xml::serde_helpers::text_content")]
    field: SomeEnum,
}

#[test]
fn text_content_round_trips() {
    let container = SomeContainer {
        field: SomeEnum::EnumValue,
    };
    let xml = "<some-container><field>EnumValue</field></some-container>";
    assert_eq!(to_string(&container).expect("serialize"), xml);
    assert_eq!(
        from_str::<SomeContainer>(xml).expect("deserialize"),
        container
    );
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct Newtype {
    #[serde(rename = "@attribute")]
    attribute: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum InternallyTagged {
    NewType(Newtype),
    Other,
}

impl_deserialize_for_internally_tagged_enum! {
    InternallyTagged, "@tag",
    ("NewType" => NewType(Newtype)),
    (_ => Other),
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct Root {
    one: InternallyTagged,
    two: InternallyTagged,
}

#[test]
fn internally_tagged_enum_macro_is_reexported() {
    assert_eq!(
        from_str::<Root>(
            r#"<root><one tag="NewType" attribute="42"/><two tag="Whatever"/></root>"#
        )
        .expect("deserialize"),
        Root {
            one: InternallyTagged::NewType(Newtype { attribute: 42 }),
            two: InternallyTagged::Other,
        }
    );
}
