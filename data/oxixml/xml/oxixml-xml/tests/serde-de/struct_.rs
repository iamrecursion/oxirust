use super::*;

/// Type where all struct fields represented by elements
#[derive(Debug, Deserialize, PartialEq)]
struct Elements {
    float: f64,
    string: String,
}

/// Type where all struct fields represented by attributes
#[derive(Debug, Deserialize, PartialEq)]
struct Attributes {
    #[serde(rename = "@float")]
    float: f64,
    #[serde(rename = "@string")]
    string: String,
}

/// Type where one field represented by an attribute and one by an element
#[derive(Debug, Deserialize, PartialEq)]
struct Mixed {
    #[serde(rename = "@float")]
    float: f64,
    string: String,
}

/// Type where one field represented by list type
#[derive(Debug, Deserialize, PartialEq)]
struct List {
    item: Vec<f64>,
}

#[test]
fn elements() {
    let data: Elements = from_str(
        // Comment for prevent unnecessary formatting - we use the same style in all tests
        r#"<root><float>42</float><string>answer</string></root>"#,
    )
    .expect("should be Ok");
    assert_eq!(
        data,
        Elements {
            float: 42.0,
            string: "answer".into()
        }
    );
}

#[test]
fn excess_elements() {
    let data: Elements = from_str(
        r#"
            <root>
                <before/>
                <float>42</float>
                <in-the-middle/>
                <string>answer</string>
                <after/>
            </root>"#,
    )
    .expect("should be Ok");
    assert_eq!(
        data,
        Elements {
            float: 42.0,
            string: "answer".into()
        }
    );
}

#[test]
fn attributes() {
    let data: Attributes = from_str(
        // Comment for prevent unnecessary formatting - we use the same style in all tests
        r#"<root float="42" string="answer"/>"#,
    )
    .expect("should be Ok");
    assert_eq!(
        data,
        Attributes {
            float: 42.0,
            string: "answer".into()
        }
    );
}

#[test]
fn excess_attributes() {
    let data: Attributes =
        from_str(r#"<root before="1" float="42" in-the-middle="2" string="answer" after="3"/>"#)
            .expect("should be Ok");
    assert_eq!(
        data,
        Attributes {
            float: 42.0,
            string: "answer".into()
        }
    );
}

#[test]
fn attribute_and_element() {
    let data: Mixed = from_str(
        r#"
            <root float="42">
                <string>answer</string>
            </root>
        "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        Mixed {
            float: 42.0,
            string: "answer".into()
        }
    );
}

#[test]
fn namespaces() {
    let data: Elements = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root xmlns:namespace="http://name.space"><namespace:float>42</namespace:float><string>answer</string></root>"#,
        )
        .expect("should be Ok");
    assert_eq!(
        data,
        Elements {
            float: 42.0,
            string: "answer".into()
        }
    );
}

/// Checks that excess data before the struct correctly handled.
/// Any data not allowed before the struct
mod excess_data_before {
    use super::*;

    /// Space-only text events does not treated as data
    #[test]
    fn text_spaces_only() {
        let data: Elements = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            " \t\n\r<root><float>42</float><string>answer</string></root>",
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Elements {
                float: 42.0,
                string: "answer".into()
            }
        );
    }

    /// Text events with non-space characters are not allowed
    #[test]
    fn text_non_spaces() {
        match from_str::<Elements>(
            "\nexcess text\t<root><float>42</float><string>answer</string></root>",
        ) {
            Err(DeError::Custom(reason)) => assert_eq!(
                reason,
                "invalid type: string \"\\nexcess text\\t\", expected struct Elements",
            ),
            x => panic!(
                r#"Expected `Err(Custom("invalid type: string \"\\nexcess text\\t\", expected struct Elements"))`, but got `{:?}`"#,
                x
            ),
        };
    }

    /// CDATA events are not allowed
    #[test]
    fn cdata() {
        match from_str::<Elements>(
            "<![CDATA[excess cdata]]><root><float>42</float><string>answer</string></root>",
        ) {
            Err(DeError::Custom(reason)) => assert_eq!(
                reason,
                "invalid type: string \"excess cdata\", expected struct Elements",
            ),
            x => panic!(
                r#"Expected `Err(Custom("invalid type: string \"excess cdata\", expected struct Elements"))`, but got `{:?}`"#,
                x
            ),
        };
    }

    /// Comments are ignored, so they are allowed
    #[test]
    fn comment() {
        let data: Elements = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            "<!--comment--><root><float>42</float><string>answer</string></root>",
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Elements {
                float: 42.0,
                string: "answer".into()
            }
        );
    }

    /// Processing instructions are ignored, so they are allowed
    #[test]
    fn pi() {
        let data: Elements = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            "<?pi?><root><float>42</float><string>answer</string></root>",
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Elements {
                float: 42.0,
                string: "answer".into()
            }
        );
    }
}

maplike_errors!(Attributes, Mixed, List);
