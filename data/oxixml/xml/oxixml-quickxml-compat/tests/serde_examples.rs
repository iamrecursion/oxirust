//! Compiles and runs the `quick-xml` 0.41 `se`/`de` module rustdoc examples
//! verbatim against `oxixml-quickxml-compat`, proving serde (de)serialization
//! path-compatibility. Requires the `serialize` feature.

use oxixml_quickxml_compat as quick_xml;
use serde::{Deserialize, Serialize};

/// `quick_xml::se::to_string` doc example.
#[test]
fn se_to_string_basic_struct() {
    use quick_xml::se::to_string;

    #[derive(Serialize)]
    struct Root<'a> {
        #[serde(rename = "@attribute")]
        attribute: &'a str,
        element: &'a str,
        #[serde(rename = "$text")]
        text: &'a str,
    }

    let data = Root {
        attribute: "attribute content",
        element: "element content",
        text: "text content",
    };

    assert_eq!(
        to_string(&data).expect("serialize"),
        // The root tag name is automatically deduced from the struct name.
        "<Root attribute=\"attribute content\">\
            <element>element content</element>\
            text content\
        </Root>"
    );
}

/// `quick_xml::se::to_string_with_root` doc example.
#[test]
fn se_to_string_with_root() {
    use quick_xml::se::to_string_with_root;

    #[derive(Serialize)]
    struct Root<'a> {
        #[serde(rename = "@attribute")]
        attribute: &'a str,
        element: &'a str,
        #[serde(rename = "$text")]
        text: &'a str,
    }

    let data = Root {
        attribute: "attribute content",
        element: "element content",
        text: "text content",
    };

    assert_eq!(
        to_string_with_root("top-level", &data).expect("serialize"),
        "<top-level attribute=\"attribute content\">\
            <element>element content</element>\
            text content\
        </top-level>"
    );
}

/// `quick_xml::de::from_str` round trip through `quick_xml::se::to_string`,
/// exercising the `de` module import path.
#[test]
fn de_from_str_round_trip() {
    use quick_xml::de::from_str;
    use quick_xml::se::to_string;

    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Item {
        #[serde(rename = "@id")]
        id: u32,
        name: String,
    }

    let xml = "<Item id=\"7\"><name>widget</name></Item>";
    let item: Item = from_str(xml).expect("deserialize");
    assert_eq!(
        item,
        Item {
            id: 7,
            name: "widget".to_string(),
        }
    );

    let round_tripped = to_string(&item).expect("serialize");
    assert_eq!(round_tripped, xml);
}

/// `quick_xml::de::from_str` deserializing a `Vec` of consecutive sibling
/// elements into a sequence field, the pattern documented under `de`'s
/// "`<element>` lists" section.
#[test]
fn de_from_str_element_list() {
    use quick_xml::de::from_str;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Items {
        item: Vec<String>,
    }

    let xml = "<Items><item>a</item><item>b</item><item>c</item></Items>";
    let items: Items = from_str(xml).expect("deserialize");
    assert_eq!(
        items,
        Items {
            item: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        }
    );
}
