//! Serialization through a [`Serializer`](oxixml_xml::se::Serializer) that
//! was given an explicit root tag name.

use super::*;

macro_rules! serialize_as {
    ($name:ident: $data:expr => $expected:literal) => {
        #[test]
        fn $name() {
            serialize_as!(@ $data => $expected);

            // Roundtrip to ensure that serializer corresponds to deserializer
            assert_eq!(
                $data,
                from_str($expected).expect("deserialization roundtrip"),
                "deserialization roundtrip",
            );
        }
    };
    ($name:ident: $data:expr ; $ty:ty => $expected:literal) => {
        #[test]
        fn $name() {
            serialize_as!(@ $data => $expected);

            // Roundtrip to ensure that serializer corresponds to deserializer
            assert_eq!(
                $data,
                from_str::<'_, $ty>($expected).expect("deserialization roundtrip"),
                "deserialization roundtrip",
            );
        }
    };
    (@ $data:expr => $expected:literal) => {
        let mut buffer = String::new();
        let ser = Serializer::with_root(&mut buffer, Some("root")).expect("serialization should succeed");

        $data.serialize(ser).expect("serialization should succeed");
        assert_eq!(buffer, $expected);
    };
}
macro_rules! serialize_as_only {
    ($name:ident: $data:expr => $expected:literal) => {
        #[test]
        fn $name() {
            serialize_as!(@ $data => $expected);
        }
    };
}

/// Checks that attempt to serialize given `$data` results to a
/// serialization error `$kind` with `$reason`
macro_rules! err {
    ($name:ident: $data:expr => $kind:ident($reason:literal)) => {
        #[test]
        fn $name() {
            let mut buffer = String::new();
            let ser = Serializer::with_root(&mut buffer, Some("root"))
                .expect("serialization should succeed");

            match $data.serialize(ser) {
                Err(SeError::$kind(e)) => assert_eq!(e, $reason),
                e => panic!(
                    "Expected `Err({}({}))`, but got `{:?}`",
                    stringify!($kind),
                    $reason,
                    e
                ),
            }
            // We can write something before fail
            // assert_eq!(buffer, "");
        }
    };
}

serialize_as!(false_: false => "<root>false</root>");
serialize_as!(true_:  true  => "<root>true</root>");

serialize_as!(i8_:    -42i8                => "<root>-42</root>");
serialize_as!(i16_:   -4200i16             => "<root>-4200</root>");
serialize_as!(i32_:   -42000000i32         => "<root>-42000000</root>");
serialize_as!(i64_:   -42000000000000i64   => "<root>-42000000000000</root>");
serialize_as!(isize_: -42000000isize       => "<root>-42000000</root>");

serialize_as!(u8_:    42u8                => "<root>42</root>");
serialize_as!(u16_:   4200u16             => "<root>4200</root>");
serialize_as!(u32_:   42000000u32         => "<root>42000000</root>");
serialize_as!(u64_:   42000000000000u64   => "<root>42000000000000</root>");
serialize_as!(usize_: 42000000usize       => "<root>42000000</root>");

serialize_as!(i128_: -420000000000000000000000000000i128 => "<root>-420000000000000000000000000000</root>");
serialize_as!(u128_:  420000000000000000000000000000u128 => "<root>420000000000000000000000000000</root>");

serialize_as!(f32_: 4.2f32 => "<root>4.2</root>");
serialize_as!(f64_: 4.2f64 => "<root>4.2</root>");

serialize_as!(char_non_escaped: 'h'  => "<root>h</root>");
serialize_as!(char_lt:          '<'  => "<root>&lt;</root>");
serialize_as!(char_gt:          '>'  => "<root>&gt;</root>");
serialize_as!(char_amp:         '&'  => "<root>&amp;</root>");
serialize_as!(char_apos:        '\'' => "<root>'</root>");
serialize_as!(char_quot:        '"'  => "<root>\"</root>");
serialize_as!(char_space:       ' '  => "<root> </root>");

serialize_as!(str_empty: ""; &str => "<root/>");
serialize_as!(str_non_escaped: "non-escaped string"; &str => "<root>non-escaped string</root>");
serialize_as!(str_escaped: "<\"escaped & string'>"; String => "<root>&lt;\"escaped &amp; string'&gt;</root>");
// NOTE: do not use \r, because it normalized to \n during deserialization
// but writes as is during serialization
serialize_as!(str_spaces_only: " \n\t"; String => "<root> \n\t</root>");

err!(bytes: Bytes(b"<\"escaped & bytes'>") => Unsupported("`serialize_bytes` not supported yet"));

serialize_as!(option_none: Option::<&str>::None => "");
serialize_as!(option_some: Some("non-escaped string") => "<root>non-escaped string</root>");
// NOTE: do not use \r, because it normalized to \n during deserialization
// but writes as is during serialization
serialize_as!(option_spaces_only: Some(" \n\t") => "<root> \n\t</root>");

serialize_as!(unit:
    ()
    => "<root/>");
serialize_as!(unit_struct:
    Unit
    => "<root/>");

serialize_as!(newtype:
    Newtype(true)
    => "<root>true</root>");

serialize_as!(seq:
    vec![1, 2, 3]; Vec<usize>
    => "<root>1</root>\
        <root>2</root>\
        <root>3</root>");
serialize_as!(tuple:
    // Use to_string() to get owned type that is required for deserialization
    // NOTE: do not use \r, because it normalized to \n during deserialization
    // but writes as is during serialization
    ("<\"&'>".to_string(), "with\t\n spaces", 3usize)
    => "<root>&lt;\"&amp;'&gt;</root>\
        <root>with\t\n spaces</root>\
        <root>3</root>");
serialize_as!(tuple_struct:
    Tuple(42.0, "answer")
    => "<root>42</root>\
        <root>answer</root>");

serialize_as!(map:
    BTreeMap::from([("$text", 1), ("_2", 3)])
    => "<root>\
            1\
            <_2>3</_2>\
        </root>");
serialize_as!(struct_:
    Struct {
        float: 42.0,
        string: "answer"
    }
    => "<root>\
            <float>42</float>\
            <string>answer</string>\
        </root>");
serialize_as!(nested_struct:
    NestedStruct {
        nested: Nested { float: 42.0 },
        string: "answer",
    }
    => "<root>\
            <nested>\
                <float>42</float>\
            </nested>\
            <string>answer</string>\
        </root>");
// NOTE: Cannot be deserialized in roundtrip due to
// https://github.com/serde-rs/serde/issues/1183
serialize_as_only!(flatten_struct:
    FlattenStruct {
        nested: Nested { float: 42.0 },
        string: "answer",
    }
    => "<root>\
            <float>42</float>\
            <string>answer</string>\
        </root>");
serialize_as!(empty_struct:
    Empty {}
    => "<root/>");
serialize_as!(text:
    Text {
        float: 42.0,
        string: "answer"
    }
    => "<root>\
            42\
            <string>answer</string>\
        </root>");

mod enum_ {
    use super::*;

    mod externally_tagged {
        use super::*;

        serialize_as!(unit:
            ExternallyTagged::Unit
            => "<Unit/>");
        serialize_as!(newtype:
            ExternallyTagged::Newtype(true)
            => "<Newtype>true</Newtype>");
        serialize_as!(tuple:
            ExternallyTagged::Tuple(42.0, "answer")
            => "<Tuple>42</Tuple>\
                <Tuple>answer</Tuple>");
        serialize_as!(struct_:
            ExternallyTagged::Struct {
                float: 42.0,
                string: "answer",
            }
            => "<Struct>\
                    <float>42</float>\
                    <string>answer</string>\
                </Struct>");
        serialize_as!(nested_struct:
            ExternallyTagged::Holder {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<Holder>\
                    <nested>\
                        <float>42</float>\
                    </nested>\
                    <string>answer</string>\
                </Holder>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(flatten_struct:
            ExternallyTagged::Flatten {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<Flatten>\
                    <float>42</float>\
                    <string>answer</string>\
                </Flatten>");
        serialize_as!(empty_struct:
            ExternallyTagged::Empty {}
            => "<Empty/>");
        serialize_as!(text:
            ExternallyTagged::Text {
                float: 42.0,
                string: "answer"
            }
            => "<Text>\
                    42\
                    <string>answer</string>\
                </Text>");

        /// Test serialization of the specially named variant `$text`
        mod text {
            use super::*;

            #[derive(Debug, PartialEq, Deserialize, Serialize)]
            enum Unit {
                #[serde(rename = "$text")]
                Text,
            }
            #[derive(Debug, PartialEq, Deserialize, Serialize)]
            enum Newtype<'a> {
                #[serde(rename = "$text")]
                Text(&'a str),
            }
            #[derive(Debug, PartialEq, Deserialize, Serialize)]
            enum Tuple {
                #[serde(rename = "$text")]
                Text(f64, String),
            }
            #[derive(Debug, PartialEq, Deserialize, Serialize)]
            enum Struct<'a> {
                #[serde(rename = "$text")]
                Text { float: f64, string: &'a str },
            }

            // It is unknown how to exactly serialize unit to a text
            err!(unit: Unit::Text => Unsupported("cannot serialize enum unit variant `Unit::$text` as text content value"));
            serialize_as!(newtype: Newtype::Text("newtype text") => "newtype text");
            // Tuple variant serialized as an `xs:list`
            serialize_as!(tuple: Tuple::Text(4.2, "newtype-text".into()) => "4.2 newtype-text");
            // Note, that spaces in strings, even escaped, would represent
            // the list item delimiters. Non-symmetric serialization follows
            // tradition: the XmlBeans Java library have the same behavior.
            // See also <https://stackoverflow.com/questions/45494204/escape-space-in-xml-xslist>
            serialize_as_only!(tuple_with_spaces: Tuple::Text(4.2, "newtype text".into()) => "4.2 newtype&#32;text");
            // Struct variant cannot be directly serialized to a text
            err!(struct_:
                Struct::Text {
                    float: 4.2,
                    string: "newtype text",
                }
                => Unsupported("cannot serialize enum struct variant `Struct::$text` as text content value"));
        }
    }

    /// Name `$text` has no special meaning in adjacently tagged enums
    mod internally_tagged {
        use super::*;

        serialize_as!(unit:
            InternallyTagged::Unit
            => "<root>\
                    <tag>Unit</tag>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(newtype:
            InternallyTagged::Newtype(Nested { float: 4.2 })
            => "<root>\
                    <tag>Newtype</tag>\
                    <float>4.2</float>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(struct_:
            InternallyTagged::Struct {
                float: 42.0,
                string: "answer",
            }
            => "<root>\
                    <tag>Struct</tag>\
                    <float>42</float>\
                    <string>answer</string>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(nested_struct:
            InternallyTagged::Holder {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<root>\
                    <tag>Holder</tag>\
                    <nested>\
                        <float>42</float>\
                    </nested>\
                    <string>answer</string>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(flatten_struct:
            InternallyTagged::Flatten {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<root>\
                    <tag>Flatten</tag>\
                    <float>42</float>\
                    <string>answer</string>\
                </root>");
        serialize_as!(empty_struct:
            InternallyTagged::Empty {}
            => "<root>\
                    <tag>Empty</tag>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(text:
            InternallyTagged::Text {
                float: 42.0,
                string: "answer"
            }
            => "<root>\
                    <tag>Text</tag>\
                    42\
                    <string>answer</string>\
                </root>");
    }

    /// Name `$text` has no special meaning in adjacently tagged enums
    mod adjacently_tagged {
        use super::*;

        serialize_as!(unit:
            AdjacentlyTagged::Unit
            => "<root>\
                    <tag>Unit</tag>\
                </root>");
        serialize_as!(newtype:
            AdjacentlyTagged::Newtype(true)
            => "<root>\
                    <tag>Newtype</tag>\
                    <content>true</content>\
                </root>");
        serialize_as!(tuple:
            AdjacentlyTagged::Tuple(42.0, "answer")
            => "<root>\
                    <tag>Tuple</tag>\
                    <content>42</content>\
                    <content>answer</content>\
                </root>");
        serialize_as!(struct_:
            AdjacentlyTagged::Struct {
                float: 42.0,
                string: "answer",
            }
            => "<root>\
                    <tag>Struct</tag>\
                    <content>\
                        <float>42</float>\
                        <string>answer</string>\
                    </content>\
                </root>");
        serialize_as!(nested_struct:
            AdjacentlyTagged::Holder {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<root>\
                    <tag>Holder</tag>\
                    <content>\
                        <nested>\
                            <float>42</float>\
                        </nested>\
                        <string>answer</string>\
                    </content>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(flatten_struct:
            AdjacentlyTagged::Flatten {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<root>\
                    <tag>Flatten</tag>\
                    <content>\
                        <float>42</float>\
                        <string>answer</string>\
                    </content>\
                </root>");
        serialize_as!(empty_struct:
            AdjacentlyTagged::Empty {}
            => "<root>\
                    <tag>Empty</tag>\
                    <content/>\
                </root>");
        serialize_as!(text:
            AdjacentlyTagged::Text {
                float: 42.0,
                string: "answer",
            }
            => "<root>\
                    <tag>Text</tag>\
                    <content>\
                        42\
                        <string>answer</string>\
                    </content>\
                </root>");
    }

    /// Name `$text` has no special meaning in untagged enums
    mod untagged {
        use super::*;

        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(unit:
            Untagged::Unit
            => "<root/>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(newtype:
            Untagged::Newtype(true)
            => "<root>true</root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(tuple:
            Untagged::Tuple(42.0, "answer")
            => "<root>42</root>\
                <root>answer</root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(struct_:
            Untagged::Struct {
                float: 42.0,
                string: "answer",
            }
            => "<root>\
                    <float>42</float>\
                    <string>answer</string>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(nested_struct:
            Untagged::Holder {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<root>\
                    <nested>\
                        <float>42</float>\
                    </nested>\
                    <string>answer</string>\
                </root>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(flatten_struct:
            Untagged::Flatten {
                nested: Nested { float: 42.0 },
                string: "answer",
            }
            => "<root>\
                    <float>42</float>\
                    <string>answer</string>\
                </root>");
        serialize_as!(empty_struct:
            Untagged::Empty {}
            => "<root/>");
        // NOTE: Cannot be deserialized in roundtrip due to
        // https://github.com/serde-rs/serde/issues/1183
        serialize_as_only!(text:
            Untagged::Text {
                float: 42.0,
                string: "answer"
            }
            => "<root>\
                    42\
                    <string>answer</string>\
                </root>");
    }
}
