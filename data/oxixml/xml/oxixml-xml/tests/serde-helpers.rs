//! Tests for the `serde_helpers` module, which mirrors `quick_xml::serde_helpers`
//! of quick-xml 0.41.

use oxixml_xml::de::from_str;
use oxixml_xml::impl_deserialize_for_internally_tagged_enum;
use oxixml_xml::se::to_string;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
enum SomeEnum {
    EnumValue,
    OtherValue,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
#[serde(rename = "some-container")]
struct SomeContainer {
    #[serde(with = "oxixml_xml::serde_helpers::text_content")]
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

    let other = "<some-container><field>OtherValue</field></some-container>";
    assert_eq!(
        from_str::<SomeContainer>(other).expect("deserialize"),
        SomeContainer {
            field: SomeEnum::OtherValue,
        }
    );
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct Root {
    one: InternallyTagged,
    two: InternallyTagged,
    three: InternallyTagged,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct Newtype {
    #[serde(rename = "@attribute")]
    attribute: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum InternallyTagged {
    Unit,
    NewType(Newtype),
    Struct {
        #[allow(dead_code)]
        attribute: u32,
        element: String,
    },
}

impl_deserialize_for_internally_tagged_enum! {
    InternallyTagged, "@tag",
    ("Unit"    => Unit),
    ("NewType" => NewType(Newtype)),
    ("Struct"  => Struct {
        #[serde(rename = "@attribute")]
        attribute: u32,
        element: String,
    }),
}

#[test]
fn internally_tagged_enum_is_deserialized() {
    assert_eq!(
        from_str::<Root>(
            r#"<root>
                 <one tag="Unit"/>
                 <two tag="NewType" attribute="42"/>
                 <three tag="Struct" attribute="7"><element>text</element></three>
               </root>"#
        )
        .expect("deserialize"),
        Root {
            one: InternallyTagged::Unit,
            two: InternallyTagged::NewType(Newtype { attribute: 42 }),
            three: InternallyTagged::Struct {
                attribute: 7,
                element: "text".to_owned(),
            },
        }
    );
}

#[test]
fn internally_tagged_enum_reports_a_missing_tag() {
    assert!(from_str::<Root>("<root><one/><two/><three/></root>").is_err());
}
