use oxixml_xml::DeError;
use oxixml_xml::de::Deserializer;
use oxixml_xml::utils::{ByteBuf, Bytes};

use serde::Deserialize;
use serde::de::IgnoredAny;

mod serde_helpers;
use serde_helpers::from_str;

/// Tests for deserializing into specially named field `$text` which represent
/// textual content of an XML element
mod text {
    use super::*;

    /// A minimal stand-in for `serde_value::Value`, which quick-xml uses as a
    /// dev-dependency. Only the variants needed by the test below are modelled;
    /// like `serde_value::Value` it is deserialized through `deserialize_any`,
    /// which is exactly what this test exercises.
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum Value {
        String(String),
        Map(std::collections::BTreeMap<Value, Value>),
    }

    impl<'de> Deserialize<'de> for Value {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct ValueVisitor;

            impl<'de> serde::de::Visitor<'de> for ValueVisitor {
                type Value = Value;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("any XML value")
                }

                fn visit_str<E>(self, v: &str) -> Result<Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(Value::String(v.to_string()))
                }

                fn visit_string<E>(self, v: String) -> Result<Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(Value::String(v))
                }

                fn visit_map<A>(self, mut map: A) -> Result<Value, A::Error>
                where
                    A: serde::de::MapAccess<'de>,
                {
                    let mut entries = std::collections::BTreeMap::new();
                    while let Some((key, value)) = map.next_entry()? {
                        entries.insert(key, value);
                    }
                    Ok(Value::Map(entries))
                }
            }

            deserializer.deserialize_any(ValueVisitor)
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/231
    #[test]
    fn implicit() {
        let item: Value = from_str(r#"<root>content</root>"#).expect("should be Ok");

        assert_eq!(
            item,
            Value::Map(
                vec![(
                    Value::String("$text".into()),
                    Value::String("content".into())
                )]
                .into_iter()
                .collect()
            )
        );
    }

    #[test]
    fn explicit() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Item {
            #[serde(rename = "$text")]
            content: String,
        }

        let item: Item = from_str(r#"<root>content</root>"#).expect("should be Ok");

        assert_eq!(
            item,
            Item {
                content: "content".into()
            }
        );
    }

    #[test]
    fn without() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Item;

        let _: Item = from_str(r#"<root>content</root>"#).expect("should be Ok");
    }
}

/// Tests calling `deserialize_ignored_any`
#[test]
fn ignored_any() {
    let err = from_str::<IgnoredAny>("");
    match err {
        Err(DeError::UnexpectedEof) => {}
        x => panic!("Expected `Err(UnexpectedEof)`, but got `{:?}`", x),
    }

    from_str::<IgnoredAny>(r#"<empty/>"#).expect("should be Ok");
    from_str::<IgnoredAny>(r#"<with-attributes key="value"/>"#).expect("should be Ok");
    from_str::<IgnoredAny>(r#"<nested>text</nested>"#).expect("should be Ok");
    from_str::<IgnoredAny>(r#"<nested><![CDATA[cdata]]></nested>"#).expect("should be Ok");
    from_str::<IgnoredAny>(r#"<nested><nested/></nested>"#).expect("should be Ok");
}

/// Tests for trivial XML documents: empty or contains only primitive type
/// on a top level; all of them should be considered invalid
#[path = "serde-de/trivial.rs"]
mod trivial;

mod unit {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Unit;

    #[test]
    fn simple() {
        let data: Unit = from_str("<root/>").expect("should be Ok");
        assert_eq!(data, Unit);
    }

    #[test]
    fn excess_attribute() {
        let data: Unit = from_str(r#"<root excess="attribute"/>"#).expect("should be Ok");
        assert_eq!(data, Unit);
    }

    #[test]
    fn excess_element() {
        let data: Unit =
            from_str(r#"<root><excess>element</excess></root>"#).expect("should be Ok");
        assert_eq!(data, Unit);
    }

    #[test]
    fn excess_text() {
        let data: Unit = from_str(r#"<root>excess text</root>"#).expect("should be Ok");
        assert_eq!(data, Unit);
    }

    #[test]
    fn excess_cdata() {
        let data: Unit =
            from_str(r#"<root><![CDATA[excess CDATA]]></root>"#).expect("should be Ok");
        assert_eq!(data, Unit);
    }
}

mod newtype {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Newtype(bool);

    #[test]
    fn simple() {
        let data: Newtype = from_str("<root>true</root>").expect("should be Ok");
        assert_eq!(data, Newtype(true));
    }

    #[test]
    fn excess_attribute() {
        let data: Newtype =
            from_str(r#"<root excess="attribute">true</root>"#).expect("should be Ok");
        assert_eq!(data, Newtype(true));
    }
}

mod tuple {
    use super::*;

    #[test]
    fn simple() {
        let data: (f32, String) = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            "<root>42</root><root>answer</root>",
        )
        .expect("should be Ok");
        assert_eq!(data, (42.0, "answer".into()));
    }

    #[test]
    fn excess_attribute() {
        let data: (f32, String) = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root excess="attribute">42</root><root>answer</root>"#,
        )
        .expect("should be Ok");
        assert_eq!(data, (42.0, "answer".into()));
    }
}

#[path = "serde-de/tuple_struct.rs"]
mod tuple_struct;

// seq tests are so big, so it in the separate file serde-de-seq.rs to speed-up compilation

macro_rules! maplike_errors {
    ($type:ty, $list:ty) => {
        maplike_errors!($type, $type, $list);
    };
    (
        $attributes:ty,
        $mixed:ty,
        $list:ty
    ) => {
        mod non_closed {
            use super::*;
            use oxixml_xml::errors::{Error, IllFormedError};

            /// For struct we expect that error about not closed tag appears
            /// earlier than error about missing fields
            #[test]
            fn missing_field() {
                let data = from_str::<$mixed>(r#"<root>"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                        assert_eq!(cause, IllFormedError::MissingEndTag("root".into()))
                    }
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }

            #[test]
            fn attributes() {
                let data = from_str::<$attributes>(r#"<root float="42" string="answer">"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                        assert_eq!(cause, IllFormedError::MissingEndTag("root".into()))
                    }
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }

            #[test]
            fn elements_root() {
                let data = from_str::<$mixed>(r#"<root float="42"><string>answer</string>"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                        assert_eq!(cause, IllFormedError::MissingEndTag("root".into()))
                    }
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }

                // Reaches `DeEvent::Eof` in `MapValueSeqAccess::next_element_seed`
                let data = from_str::<$list>(r#"<root><item>1</item><item>2</item>"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                        assert_eq!(cause, IllFormedError::MissingEndTag("root".into()))
                    }
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }

            #[test]
            fn elements_child() {
                // Reaches `Deserializer::read_text` when text is not present
                let data = from_str::<$mixed>(r#"<root float="42"><string>"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                        assert_eq!(cause, IllFormedError::MissingEndTag("string".into()))
                    }
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }

                // Reaches `Deserializer::read_text` when text is present
                let data = from_str::<$mixed>(r#"<root float="42"><string>answer"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                        assert_eq!(cause, IllFormedError::MissingEndTag("string".into()))
                    }
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }
        }

        mod mismatched_end {
            use super::*;
            use oxixml_xml::errors::{Error, IllFormedError};

            /// For struct we expect that error about mismatched tag appears
            /// earlier than error about missing fields
            #[test]
            fn missing_field() {
                let data = from_str::<$mixed>(r#"<root></mismatched>"#);

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => assert_eq!(
                        cause,
                        IllFormedError::MismatchedEndTag {
                            expected: "root".into(),
                            found: "mismatched".into(),
                        }
                    ),
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }

            #[test]
            fn attributes() {
                let data = from_str::<$attributes>(
                    // Comment for prevent unnecessary formatting - we use the same style in all tests
                    r#"<root float="42" string="answer"></mismatched>"#,
                );

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => assert_eq!(
                        cause,
                        IllFormedError::MismatchedEndTag {
                            expected: "root".into(),
                            found: "mismatched".into(),
                        }
                    ),
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }

            #[test]
            fn elements_root() {
                let data = from_str::<$mixed>(
                    // Comment for prevent unnecessary formatting - we use the same style in all tests
                    r#"<root float="42"><string>answer</string></mismatched>"#,
                );

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => assert_eq!(
                        cause,
                        IllFormedError::MismatchedEndTag {
                            expected: "root".into(),
                            found: "mismatched".into(),
                        }
                    ),
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }

            #[test]
            fn elements_child() {
                let data = from_str::<$mixed>(
                    // Comment for prevent unnecessary formatting - we use the same style in all tests
                    r#"<root float="42"><string>answer</mismatched></root>"#,
                );

                match data {
                    Err(DeError::InvalidXml(Error::IllFormed(cause))) => assert_eq!(
                        cause,
                        IllFormedError::MismatchedEndTag {
                            expected: "string".into(),
                            found: "mismatched".into(),
                        }
                    ),
                    x => panic!(
                        "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                        x
                    ),
                }
            }
        }

        mod incomplete_tag {
            use super::*;
            use oxixml_xml::errors::{Error, SyntaxError};

            #[test]
            fn attributes() {
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                let xml = r#"<root float="42" string="answer""#;
                let mut de = Deserializer::from_str(xml);
                let err = <$attributes>::deserialize(&mut de).expect_err("should be Err");
                match err {
                    DeError::InvalidXml(Error::Syntax(SyntaxError::UnclosedTag)) => (),
                    _ => panic!(
                        "Expected `Err(InvalidXml(Syntax(UnclosedTag)))`, but got `{:?}`",
                        err
                    ),
                }
                // The unclosed tag starts at the very beginning of the document.
                let reader = de.get_ref().get_ref();
                assert_eq!(reader.error_position(), 0);
                assert_eq!(reader.buffer_position(), xml.len() as u64);
            }

            #[test]
            fn elements_root() {
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                let xml = r#"<root float="42"><string>answer</string><root"#;
                let mut de = Deserializer::from_str(xml);
                let err = <$mixed>::deserialize(&mut de).expect_err("should be Err");
                match err {
                    DeError::InvalidXml(Error::Syntax(SyntaxError::UnclosedTag)) => (),
                    _ => panic!(
                        "Expected `Err(InvalidXml(Syntax(UnclosedTag)))`, but got `{:?}`",
                        err
                    ),
                }
                // The unclosed tag is the second `<root`, not the first.
                let reader = de.get_ref().get_ref();
                let expected = xml.rfind("<root").expect("test fixture contains `<root`");
                assert_eq!(reader.error_position(), expected as u64);
                assert_eq!(reader.buffer_position(), xml.len() as u64);
            }

            #[test]
            fn elements_child() {
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                let xml = r#"<root float="42"><string>answer</string"#;
                let mut de = Deserializer::from_str(xml);
                let err = <$mixed>::deserialize(&mut de).expect_err("should be Err");
                match err {
                    DeError::InvalidXml(Error::Syntax(SyntaxError::UnclosedTag)) => (),
                    _ => panic!(
                        "Expected `Err(InvalidXml(Syntax(UnclosedTag)))`, but got `{:?}`",
                        err
                    ),
                }
                // The unclosed tag is the truncated `</string` closing tag.
                let reader = de.get_ref().get_ref();
                let expected = xml
                    .rfind("</string")
                    .expect("test fixture contains `</string`");
                assert_eq!(reader.error_position(), expected as u64);
                assert_eq!(reader.buffer_position(), xml.len() as u64);
            }
        }
    };
}

mod map {
    use super::*;
    use std::collections::HashMap;
    use std::iter::FromIterator;

    #[test]
    fn elements() {
        let data: HashMap<(), ()> = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root><float>42</float><string>answer</string></root>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            HashMap::from_iter([((), ()), ((), ()),].iter().cloned())
        );
    }

    #[test]
    fn attributes() {
        let data: HashMap<(), ()> = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root float="42" string="answer"/>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            HashMap::from_iter([((), ()), ((), ()),].iter().cloned())
        );
    }

    #[test]
    fn attribute_and_element() {
        let data: HashMap<(), ()> = from_str(
            r#"
            <root float="42">
                <string>answer</string>
            </root>
            "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            HashMap::from_iter([((), ()), ((), ()),].iter().cloned())
        );
    }

    maplike_errors!(HashMap<(), ()>, HashMap<(), Vec<u32>>);
}

#[path = "serde-de/struct_.rs"]
mod struct_;

mod nested_struct {
    use super::*;

    #[test]
    fn elements() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Struct {
            nested: Nested,
            string: String,
        }

        #[derive(Debug, Deserialize, PartialEq)]
        struct Nested {
            float: f32,
        }

        let data: Struct = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root><string>answer</string><nested><float>42</float></nested></root>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Struct {
                nested: Nested { float: 42.0 },
                string: "answer".into()
            }
        );
    }

    #[test]
    fn attributes() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Struct {
            nested: Nested,
            #[serde(rename = "@string")]
            string: String,
        }

        #[derive(Debug, Deserialize, PartialEq)]
        struct Nested {
            #[serde(rename = "@float")]
            float: f32,
        }

        let data: Struct = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root string="answer"><nested float="42"/></root>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Struct {
                nested: Nested { float: 42.0 },
                string: "answer".into()
            }
        );
    }
}

mod flatten_struct {
    use super::*;

    #[test]
    fn elements() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Struct {
            #[serde(flatten)]
            nested: Nested,
            string: String,
        }

        #[derive(Debug, Deserialize, PartialEq)]
        struct Nested {
            // Kept as `String` rather than `f64`: `#[serde(flatten)]` routes
            // through `deserialize_any`, which surfaces a text-only element as
            // its string content but has no way to know it should reparse that
            // text as an `f64` (serde-rs/serde#1183). A `String` field accepts
            // the value; an `f64` field could not be recovered from the buffered
            // string.
            float: String,
        }

        let data: Struct = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root><float>42</float><string>answer</string></root>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Struct {
                nested: Nested { float: "42".into() },
                string: "answer".into()
            }
        );
    }

    #[test]
    fn attributes() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Struct {
            #[serde(flatten)]
            nested: Nested,
            #[serde(rename = "@string")]
            string: String,
        }

        #[derive(Debug, Deserialize, PartialEq)]
        struct Nested {
            // Kept as `String` rather than `f64`: `#[serde(flatten)]` routes
            // through `deserialize_any` regardless of the field's declared
            // type or whether the source is an attribute or an element, so it
            // always sees a string here too. Confirmed against the real
            // upstream `quick-xml` 0.41.0 crate, which fails identically.
            // Blocked on serde-rs/serde#1183, not on anything in this crate.
            #[serde(rename = "@float")]
            float: String,
        }

        let data: Struct = from_str(
            // Comment for prevent unnecessary formatting - we use the same style in all tests
            r#"<root float="42" string="answer"/>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            Struct {
                nested: Nested { float: "42".into() },
                string: "answer".into()
            }
        );
    }
}

// enum tests are so big, so it in the separate file serde-de-seq.rs to speed-up compilation

/// https://www.w3schools.com/xml/el_list.asp
mod xml_schema_lists {
    use super::*;

    macro_rules! list {
        ($name:ident: $type:ty = $xml:literal => $result:expr) => {
            #[test]
            fn $name() {
                let data: List<$type> = from_str($xml).expect("should be Ok");

                assert_eq!(data, List { list: $result });
            }
        };
    }

    macro_rules! err {
        ($name:ident: $type:ty = $xml:literal => $kind:ident($err:literal)) => {
            #[test]
            fn $name() {
                let err = from_str::<List<$type>>($xml).expect_err("should be Err");

                match err {
                    DeError::$kind(e) => assert_eq!(e, $err),
                    _ => panic!(
                        "Expected `Err({}({}))`, but got `{:?}`",
                        stringify!($kind),
                        $err,
                        err
                    ),
                }
            }
        };
    }

    /// Checks that sequences can be deserialized from an XML attribute content
    /// according to the `xs:list` XML Schema type
    mod attribute {
        use super::*;

        #[derive(Debug, Deserialize, PartialEq)]
        struct List<T> {
            #[serde(rename = "@list")]
            list: Vec<T>,
        }

        list!(i8_:  i8  = r#"<root list="1 -2  3"/>"# => vec![1, -2, 3]);
        list!(i16_: i16 = r#"<root list="1 -2  3"/>"# => vec![1, -2, 3]);
        list!(i32_: i32 = r#"<root list="1 -2  3"/>"# => vec![1, -2, 3]);
        list!(i64_: i64 = r#"<root list="1 -2  3"/>"# => vec![1, -2, 3]);

        list!(u8_:  u8  = r#"<root list="1 2  3"/>"# => vec![1, 2, 3]);
        list!(u16_: u16 = r#"<root list="1 2  3"/>"# => vec![1, 2, 3]);
        list!(u32_: u32 = r#"<root list="1 2  3"/>"# => vec![1, 2, 3]);
        list!(u64_: u64 = r#"<root list="1 2  3"/>"# => vec![1, 2, 3]);

        list!(i128_: i128 = r#"<root list="1 -2  3"/>"# => vec![1, -2, 3]);
        list!(u128_: u128 = r#"<root list="1 2  3"/>"# => vec![1, 2, 3]);

        list!(f32_: f32 = r#"<root list="1.23 -4.56  7.89"/>"# => vec![1.23, -4.56, 7.89]);
        list!(f64_: f64 = r#"<root list="1.23 -4.56  7.89"/>"# => vec![1.23, -4.56, 7.89]);

        list!(bool_: bool = r#"<root list="true false  true"/>"# => vec![true, false, true]);
        list!(char_: char = r#"<root list="4 2  j"/>"# => vec!['4', '2', 'j']);

        list!(string: String = r#"<root list="first second  third&#x20;3"/>"# => vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
            "3".to_string(),
        ]);
        err!(byte_buf: ByteBuf = r#"<root list="first second  third&#x20;3"/>"#
            => Custom("invalid type: string \"first\", expected byte data"));

        list!(unit: () = r#"<root list="1 second  false"/>"# => vec![(), (), ()]);
    }

    /// Checks that sequences can be deserialized from an XML text content
    /// according to the `xs:list` XML Schema type
    mod element {
        use super::*;

        #[derive(Debug, Deserialize, PartialEq)]
        struct List<T> {
            // Give it a special name that means text content of the XML node
            #[serde(rename = "$text")]
            list: Vec<T>,
        }

        mod text {
            use super::*;

            list!(i8_:  i8  = "<root>1 -2  3</root>" => vec![1, -2, 3]);
            list!(i16_: i16 = "<root>1 -2  3</root>" => vec![1, -2, 3]);
            list!(i32_: i32 = "<root>1 -2  3</root>" => vec![1, -2, 3]);
            list!(i64_: i64 = "<root>1 -2  3</root>" => vec![1, -2, 3]);

            list!(u8_:  u8  = "<root>1 2  3</root>" => vec![1, 2, 3]);
            list!(u16_: u16 = "<root>1 2  3</root>" => vec![1, 2, 3]);
            list!(u32_: u32 = "<root>1 2  3</root>" => vec![1, 2, 3]);
            list!(u64_: u64 = "<root>1 2  3</root>" => vec![1, 2, 3]);

            list!(i128_: i128 = "<root>1 -2  3</root>" => vec![1, -2, 3]);
            list!(u128_: u128 = "<root>1 2  3</root>" => vec![1, 2, 3]);

            list!(f32_: f32 = "<root>1.23 -4.56  7.89</root>" => vec![1.23, -4.56, 7.89]);
            list!(f64_: f64 = "<root>1.23 -4.56  7.89</root>" => vec![1.23, -4.56, 7.89]);

            list!(bool_: bool = "<root>true false  true</root>" => vec![true, false, true]);
            list!(char_: char = "<root>4 2  j</root>" => vec!['4', '2', 'j']);

            // Expanding of entity references happens before list parsing
            // This is confirmed by XmlBeans (mature Java library) as well
            list!(string: String = "<root>first second  third&#x20;3</root>" => vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
                "3".to_string(),
            ]);
            err!(byte_buf: ByteBuf = "<root>first second  third&#x20;3</root>"
                => Custom("invalid type: string \"first\", expected byte data"));

            list!(unit: () = "<root>1 second  false</root>" => vec![(), (), ()]);
        }

        mod cdata {
            use super::*;

            list!(i8_:  i8  = "<root><![CDATA[1 -2  3]]></root>" => vec![1, -2, 3]);
            list!(i16_: i16 = "<root><![CDATA[1 -2  3]]></root>" => vec![1, -2, 3]);
            list!(i32_: i32 = "<root><![CDATA[1 -2  3]]></root>" => vec![1, -2, 3]);
            list!(i64_: i64 = "<root><![CDATA[1 -2  3]]></root>" => vec![1, -2, 3]);

            list!(u8_:  u8  = "<root><![CDATA[1 2  3]]></root>" => vec![1, 2, 3]);
            list!(u16_: u16 = "<root><![CDATA[1 2  3]]></root>" => vec![1, 2, 3]);
            list!(u32_: u32 = "<root><![CDATA[1 2  3]]></root>" => vec![1, 2, 3]);
            list!(u64_: u64 = "<root><![CDATA[1 2  3]]></root>" => vec![1, 2, 3]);

            list!(i128_: i128 = "<root><![CDATA[1 -2  3]]></root>" => vec![1, -2, 3]);
            list!(u128_: u128 = "<root><![CDATA[1 2  3]]></root>" => vec![1, 2, 3]);

            list!(f32_: f32 = "<root><![CDATA[1.23 -4.56  7.89]]></root>" => vec![1.23, -4.56, 7.89]);
            list!(f64_: f64 = "<root><![CDATA[1.23 -4.56  7.89]]></root>" => vec![1.23, -4.56, 7.89]);

            list!(bool_: bool = "<root><![CDATA[true false  true]]></root>" => vec![true, false, true]);
            list!(char_: char = "<root><![CDATA[4 2  j]]></root>" => vec!['4', '2', 'j']);

            // Cannot get whitespace in the value in any way if CDATA used:
            // - literal spaces means list item delimiters
            // - escaped sequences are not decoded in CDATA
            list!(string: String = "<root><![CDATA[first second  third&#x20;3]]></root>" => vec![
                "first".to_string(),
                "second".to_string(),
                "third&#x20;3".to_string(),
            ]);
            err!(byte_buf: ByteBuf = "<root>first second  third&#x20;3</root>"
                => Custom("invalid type: string \"first\", expected byte data"));

            list!(unit: () = "<root>1 second  false</root>" => vec![(), (), ()]);
        }
    }
}

/// Test for https://github.com/tafia/quick-xml/issues/324
#[test]
fn from_str_should_ignore_encoding() {
    let xml = r#"
        <?xml version="1.0" encoding="windows-1252" ?>
        <A a="€" />
    "#;

    #[derive(Debug, PartialEq, Deserialize)]
    struct A {
        #[serde(rename = "@a")]
        a: String,
    }

    let a: A = from_str(xml).expect("should be Ok");
    assert_eq!(
        a,
        A {
            a: "€".to_string()
        }
    );
}

/// Checks that deserializer is able to borrow data from the input
mod borrow {
    use super::*;
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    /// Struct that should borrow input to be able to deserialize successfully.
    /// serde implicitly borrow `&str` and `&[u8]` even without `#[serde(borrow)]`
    #[derive(Debug, Deserialize, PartialEq)]
    struct BorrowedElement<'a> {
        string: &'a str,
    }

    /// Struct that should borrow input to be able to deserialize successfully.
    /// serde implicitly borrow `&str` and `&[u8]` even without `#[serde(borrow)]`
    #[derive(Debug, Deserialize, PartialEq)]
    struct BorrowedAttribute<'a> {
        #[serde(rename = "@string")]
        string: &'a str,
    }

    /// Deserialization of all XML's in that module expected to pass because
    /// unescaping is not required, so deserialized `Borrowed*` types can hold
    /// references to the original buffer with an XML text
    mod non_escaped {
        use super::*;

        #[test]
        fn top_level() {
            let data: &str =
                from_str(r#"<root>without escape sequences</root>"#).expect("should be Ok");
            assert_eq!(data, "without escape sequences",);
        }

        #[test]
        fn element() {
            let data: BorrowedElement = from_str(
                r#"
                <root>
                    <string>without escape sequences</string>
                </root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                BorrowedElement {
                    string: "without escape sequences",
                }
            );
        }

        #[test]
        fn attribute() {
            let data: BorrowedAttribute =
                from_str(r#"<root string="without escape sequences"/>"#).expect("should be Ok");
            assert_eq!(
                data,
                BorrowedAttribute {
                    string: "without escape sequences",
                }
            );
        }
    }

    /// Deserialization of all XML's in that module expected to fail because
    /// values requires unescaping that will lead to allocating an internal
    /// buffer by deserializer, but the `Borrowed*` types couldn't take ownership
    /// on it.
    ///
    /// The same behavior demonstrates the `serde_json` crate
    mod escaped {
        use super::*;

        #[test]
        fn top_level() {
            match from_str::<&str>(r#"<root>with escape sequence: &lt;</root>"#) {
                Err(DeError::Custom(reason)) => assert_eq!(
                    reason,
                    "invalid type: string \"with escape sequence: <\", expected a borrowed string"
                ),
                e => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"with escape sequence: <\", expected a borrowed string"))`, but got `{:?}`"#,
                    e
                ),
            }
        }

        #[test]
        fn element() {
            match from_str::<BorrowedElement>(
                r#"
                <root>
                    <string>with escape sequence: &lt;</string>
                </root>"#,
            ) {
                Err(DeError::Custom(reason)) => assert_eq!(
                    reason,
                    "invalid type: string \"with escape sequence: <\", expected a borrowed string"
                ),
                e => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"with escape sequence: <\", expected a borrowed string"))`, but got `{:?}`"#,
                    e
                ),
            }
        }

        #[test]
        fn attribute() {
            match from_str::<BorrowedAttribute>(
                r#"<root string="with &quot;escape&quot; sequences"/>"#,
            ) {
                Err(DeError::Custom(reason)) => assert_eq!(
                    reason,
                    "invalid type: string \"with \\\"escape\\\" sequences\", expected a borrowed string"
                ),
                e => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"with \"escape\" sequences\", expected a borrowed string"))`, but got `{:?}`"#,
                    e
                ),
            }
        }
    }

    #[test]
    fn element_name() {
        let data: BTreeMap<&str, &str> = from_str(
            r#"
            <root>
                <element>element content</element>
                text content
            </root>"#,
        )
        .expect("should be Ok");
        assert_eq!(
            data,
            BTreeMap::from_iter([
                // Comment to prevent formatting in one line
                ("element", "element content"),
                ("$text", "\n                text content\n            "),
            ])
        );
    }
}

/// Test for entity resolver
mod resolve {
    use super::*;
    use oxixml_xml::de::EntityResolver;
    use oxixml_xml::events::BytesText;
    use std::collections::BTreeMap;
    use std::convert::Infallible;
    use std::iter::FromIterator;

    struct TestEntityResolver {
        capture_called: bool,
    }

    impl EntityResolver for TestEntityResolver {
        type Error = Infallible;

        fn capture(&mut self, doctype: BytesText) -> Result<(), Self::Error> {
            self.capture_called = true;

            assert_eq!(doctype.as_ref(), br#"dict[ <!ENTITY unc "unclassified"> ]"#);

            Ok(())
        }

        fn resolve(&self, entity: &str) -> Option<&str> {
            assert!(
                self.capture_called,
                "`EntityResolver::capture` should be called before `EntityResolver::resolve`"
            );
            match entity {
                "t1" => Some("test_one"),
                "t2" => Some("test_two"),
                _ => None,
            }
        }
    }

    #[test]
    fn resolve_custom_entity() {
        let resolver = TestEntityResolver {
            capture_called: false,
        };
        let mut de = Deserializer::with_resolver(
            br#"
            <!DOCTYPE dict[ <!ENTITY unc "unclassified"> ]>

            <root>
                <entity_one>&t1;</entity_one>
                <entity_two>&t2;</entity_two>
                <entity_three>non-entity</entity_three>
            </root>
            "#
            .as_ref(),
            resolver,
        );

        let data: BTreeMap<String, String> = BTreeMap::deserialize(&mut de).expect("should be Ok");
        assert_eq!(
            data,
            BTreeMap::from_iter([
                (String::from("entity_one"), String::from("test_one")),
                (String::from("entity_two"), String::from("test_two")),
                (String::from("entity_three"), String::from("non-entity")),
            ])
        );
    }
}

/// A DOCTYPE declaration inside an element's text content used to split
/// the surrounding character data into two consecutive `DeEvent::Text`
/// events, tripping an `unreachable!()` in `read_text` ("Cannot be two
/// consequent Text events"). The reader emits Text/DocType/Text for an
/// input like `<a>x<!DOCTYPE y>z</a>`, and `drain_text` did not drain
/// across DocType events, so the two text runs reached the
/// deserializer un-merged. Discovered via libFuzzer on a real-world
/// SAML response harness.
///
/// The expected behavior is: DOCTYPE inside an element is captured by
/// the entity resolver (it has the same role as a prolog-level DOCTYPE
/// for entity definitions) and the surrounding text is treated as one
/// continuous run.
mod doctype_in_element_text {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Wrapper {
        #[serde(rename = "$text")]
        text: String,
    }

    /// Minimal regression repro: a single DOCTYPE event between two
    /// adjacent text runs inside an element should not panic.
    #[test]
    fn single_doctype_between_text() {
        let xml = r#"<a>x<!DOCTYPE y>z</a>"#;
        let got: Wrapper = from_str(xml).expect("should be Ok");
        assert_eq!(got, Wrapper { text: "xz".into() });
    }

    /// Multiple DOCTYPE events stacked between text runs (matching the
    /// shape of the fuzzer-discovered input that originally surfaced
    /// this panic on real SAML traffic).
    #[test]
    fn multiple_doctypes_between_text() {
        let xml = r#"<a>x<!DOCTYPE y><!DOCTYPE z>w</a>"#;
        let got: Wrapper = from_str(xml).expect("should be Ok");
        assert_eq!(got, Wrapper { text: "xw".into() });
    }

    /// DOCTYPE followed by text only (no leading text). Should produce
    /// the trailing text without panic.
    #[test]
    fn leading_doctype_then_text() {
        let xml = r#"<a><!DOCTYPE y>x</a>"#;
        let got: Wrapper = from_str(xml).expect("should be Ok");
        assert_eq!(got, Wrapper { text: "x".into() });
    }

    /// Whitespace adjacent to the DOCTYPE on both sides should be
    /// preserved verbatim when the two text runs are merged — the
    /// DOCTYPE is treated as transparent, not as a delimiter.
    #[test]
    fn whitespace_around_doctype() {
        let xml = "<a>x <!DOCTYPE y>\tz</a>";
        let got: Wrapper = from_str(xml).expect("should be Ok");
        assert_eq!(
            got,
            Wrapper {
                text: "x \tz".into()
            }
        );

        // Whitespace at the element boundaries (before the leading
        // text and after the trailing text) is also preserved.
        let xml = "<a> x<!DOCTYPE y>z </a>";
        let got: Wrapper = from_str(xml).expect("should be Ok");
        assert_eq!(
            got,
            Wrapper {
                text: " xz ".into()
            }
        );
    }
}

/// Tests for https://github.com/tafia/quick-xml/pull/603.
///
/// According to <https://www.w3.org/TR/xml11/#NT-prolog> comments,
/// processing instructions and spaces are possible after XML declaration or DTD.
/// Their existence should not break deserializing
///
/// ```text
/// [22] prolog ::= XMLDecl Misc* (doctypedecl Misc*)?
/// [27] Misc   ::= Comment | PI | S
/// ```
mod xml_prolog {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn spaces() {
        assert_eq!(
            from_str::<HashMap<(), ()>>(
                r#"
        <?xml version="1.1"?>

        <!DOCTYPE dict>

        <doc>
        </doc>
        "#
            )
            .expect("should be Ok"),
            HashMap::new()
        );
    }

    #[test]
    fn comments() {
        assert_eq!(
            from_str::<HashMap<(), ()>>(
                r#"
        <?xml version="1.1"?>
        <!-- comment between xml declaration and doctype -->
        <!-- another comment -->
        <!DOCTYPE dict>
        <!-- comment between doctype and root element -->
        <!-- another comment -->
        <doc>
        </doc>
        "#,
            )
            .expect("should be Ok"),
            HashMap::new()
        );
    }

    #[test]
    fn pi() {
        assert_eq!(
            from_str::<HashMap<(), ()>>(
                r#"
        <?xml version="1.1"?>
        <?pi?>
        <?another pi?>
        <!DOCTYPE dict>
        <?pi?>
        <?another pi?>
        <doc>
        </doc>
        "#,
            )
            .expect("should be Ok"),
            HashMap::new()
        );
    }
}

/// Tests for https://github.com/tafia/quick-xml/pull/937.
///
/// Checks that correct EOL normalization rules is applied to the texts.
mod xml_version {
    use super::*;
    use oxixml_xml::errors::{Error, IllFormedError};

    #[derive(Debug, Deserialize, PartialEq)]
    struct Root {
        #[serde(rename = "@attribute")]
        attribute: String,

        #[serde(rename = "$text")]
        text: String,
    }

    #[test]
    fn v1_0_implicit() {
        assert_eq!(
            from_str::<Root>(
                "\
                <root attribute='\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}'>\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}</root>\
                "
            )
            .expect("should be Ok"),
            Root {
                attribute: " , , , \u{0085},\u{0085},\u{2028}".to_string(),
                text: "\n,\n,\n,\n\u{0085},\u{0085},\u{2028}".to_string(),
            }
        );
    }

    #[test]
    fn v1_0_explicit() {
        assert_eq!(
            from_str::<Root>(
                "\
                <?xml version='1.0'?>\
                <root attribute='\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}'>\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}</root>\
                "
            )
            .expect("should be Ok"),
            Root {
                attribute: " , , , \u{0085},\u{0085},\u{2028}".to_string(),
                text: "\n,\n,\n,\n\u{0085},\u{0085},\u{2028}".to_string(),
            }
        );
    }

    #[test]
    fn v1_1() {
        assert_eq!(
            from_str::<Root>(
                "\
                <?xml version='1.1'?>\
                <root attribute='\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}'>\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}</root>\
                "
            )
            .expect("should be Ok"),
            Root {
                attribute: " , , , , , ".to_string(),
                text: "\n,\n,\n,\n,\n,\n".to_string(),
            }
        );
    }

    #[test]
    fn unknown() {
        match from_str::<Root>(
            "\
                <?xml version='1.2'?>\
                <root attribute='\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}'>\r\n,\n,\r,\r\u{0085},\u{0085},\u{2028}</root>\
                ",
        ) {
            Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                assert_eq!(cause, IllFormedError::UnknownVersion,)
            }
            x => panic!("Expected `Err(InvalidXml(IllFormed(_)))`, but got {:?}", x),
        }
    }
}

/// Pins the documented quick-xml 0.41 parity limitation of `de::map`: the byte
/// offsets carried by an attribute error are counted from the start of the tag
/// name, not from the start of the document, so they do not move when the tag is
/// preceded by other content.
#[test]
fn attribute_error_position_is_tag_relative() {
    #[derive(Debug, Deserialize)]
    struct Root {
        #[serde(rename = "@a")]
        #[allow(dead_code)]
        a: String,
    }

    let describe = |xml: &str| match from_str::<Root>(xml) {
        Err(error) => format!("{error:?}"),
        Ok(_) => panic!("`{xml}` must not deserialize"),
    };

    // `7` is the offset of `unquoted` inside `root a=unquoted`, and stays `7`
    // however far into the document the tag itself is.
    assert_eq!(
        describe("<root a=unquoted/>"),
        "InvalidXml(InvalidAttr(UnquotedValue(7)))"
    );
    assert_eq!(
        describe("<!-- padding padding padding --><root a=unquoted/>"),
        "InvalidXml(InvalidAttr(UnquotedValue(7)))"
    );
}

/// A `$value`-typed primitive field expects the position it occupies to hold
/// text (or be directly wrapped in one level of tag), not a nested element.
/// When the parent has other content mapped to that field and the next thing
/// in the document is itself a tag, deserialization must fail with an ordinary
/// error rather than panicking.
#[test]
fn value_field_rejects_nested_element() {
    #[derive(Debug, Deserialize)]
    struct Root {
        #[serde(rename = "$value")]
        #[allow(dead_code)]
        field: i32,
    }

    match from_str::<Root>("<root><child>1</child></root>") {
        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"child"),
        x => panic!(
            "Expected `Err(UnexpectedStart(\"child\"))`, but got `{:?}`",
            x
        ),
    }
}

/// Pins a documented `quick-xml` 0.41 parity limitation: a field with a fixed
/// name only merges a single run of consecutive text/CDATA content, so text
/// interrupted by a nested element does not get concatenated into one string
/// -- it fails with `UnexpectedStart` at the interrupting element, matching
/// the real upstream `quick-xml` 0.41.0 crate exactly (see
/// `MapValueDeserializer::read_string` in `src/de/map.rs`).
#[test]
fn string_field_rejects_element_interrupting_text_run() {
    #[derive(Debug, Deserialize)]
    struct Root {
        #[allow(dead_code)]
        tag: String,
    }

    match from_str::<Root>("<root><tag>before<nested>x</nested>after</tag></root>") {
        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"nested"),
        x => panic!(
            "Expected `Err(UnexpectedStart(\"nested\"))`, but got `{:?}`",
            x
        ),
    }
}
