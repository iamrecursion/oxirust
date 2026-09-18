use super::*;

#[rustfmt::skip] // excess spaces used for readability
    macro_rules! eof {
        ($name:ident: $type:ty = $value:expr) => {
            #[test]
            fn $name() {
                match from_str::<$type>($value) {
                    Err(DeError::UnexpectedEof) => {}
                    x => panic!(
                        r#"Expected `Err(UnexpectedEof)`, but got `{:?}`"#,
                        x
                    ),
                }
            }
        };
        ($value:expr) => {
            eof!(i8_:    i8    = $value);
            eof!(i16_:   i16   = $value);
            eof!(i32_:   i32   = $value);
            eof!(i64_:   i64   = $value);
            eof!(isize_: isize = $value);

            eof!(u8_:    u8    = $value);
            eof!(u16_:   u16   = $value);
            eof!(u32_:   u32   = $value);
            eof!(u64_:   u64   = $value);
            eof!(usize_: usize = $value);

            eof!(u128_: u128 = $value);
            eof!(i128_: i128 = $value);

            eof!(f32_: f32 = $value);
            eof!(f64_: f64 = $value);

            eof!(false_: bool = $value);
            eof!(true_: bool = $value);
            eof!(char_: char = $value);

            eof!(string: String = $value);

            /// XML does not able to store binary data
            #[test]
            fn byte_buf() {
                match from_str::<ByteBuf>($value) {
                    Err(DeError::UnexpectedEof) => {}
                    x => panic!(
                        r#"Expected `Err(UnexpectedEof)`, but got `{:?}`"#,
                        x
                    ),
                }
            }

            /// XML does not able to store binary data
            #[test]
            fn bytes() {
                match from_str::<Bytes>($value) {
                    Err(DeError::UnexpectedEof) => {}
                    x => panic!(
                        r#"Expected `Err(UnexpectedEof)`, but got `{:?}`"#,
                        x
                    ),
                }
            }

            #[test]
            fn unit() {
                match from_str::<()>($value) {
                    Err(DeError::UnexpectedEof) => {}
                    x => panic!(
                        r#"Expected `Err(UnexpectedEof)`, but got `{:?}`"#,
                        x
                    ),
                }
            }
        };
    }

/// Empty document should considered invalid no matter what type we try to deserialize
mod empty_doc {
    use super::*;

    eof!("");
}

/// Document that contains only comment should be handled as if it is empty
mod only_comment {
    use super::*;

    eof!("<!--comment-->");
}

macro_rules! in_struct {
        // string, char and bool have some specifics in some tests
        (string: $type:ty = $value:expr, $expected:literal) => {
            in_struct!(
                string: $type = $value, $expected.into(), concat!(" \n\t", $expected, " \n\t").into();

                /// Try to deserialize type from a root tag without data or with only spaces
                #[test]
                fn wrapped_empty() {
                    let item: $type = from_str("<root/>").expect("should be Ok");
                    let expected: $type = "".into();
                    assert_eq!(item, expected);

                    let item: $type = from_str("<root> \r\n\t</root>").expect("should be Ok");
                    // \r\n normalized to \n
                    let expected: $type = " \n\t".into();
                    assert_eq!(item, expected);
                }

                #[test]
                fn field_empty() {
                    let item: Field<$type> = from_str("<root><value/></root>").expect("should be Ok");
                    assert_eq!(item, Field { value: "".into() });

                    let item: Field<$type> = from_str("<root><value> \r\n\t</value></root>").expect("should be Ok");
                    // \r\n normalized to \n
                    assert_eq!(item, Field { value: " \n\t".into() });
                }
            );
        };
        (char_: $type:ty = $value:expr, $expected:expr) => {
            in_struct!(
                char_: $type = $value, $expected, $expected;

                /// Try to deserialize type from a root tag without data or with only spaces
                #[test]
                fn wrapped_empty() {
                    match from_str::<$type>("<root/>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid value: string "", expected a character"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<$type>("<root> \r\n\t</root>") {
                        // \r\n normalized to \n
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid value: string " \n\t", expected a character"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }

                #[test]
                fn field_empty() {
                    match from_str::<Field<$type>>("<root><value/></root>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid value: string "", expected a character"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<Field<$type>>("<root><value> \r\n\t</value></root>") {
                        // \r\n normalized to \n
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid value: string " \n\t", expected a character"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }
            );
        };
        ($name:ident: bool = $value:expr, $expected:expr) => {
            in_struct!(
                $name: bool = $value, $expected, $expected;

                /// Try to deserialize type from a root tag without data or with only spaces
                #[test]
                fn wrapped_empty() {
                    match from_str::<bool>("<root/>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid type: string "", expected a boolean"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<bool>("<root> \r\n\t</root>") {
                        // \r\n normalized to \n
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid type: string " \n\t", expected a boolean"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }

                #[test]
                fn field_empty() {
                    match from_str::<Field<bool>>("<root><value/></root>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid type: string "", expected a boolean"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<Field<bool>>("<root><value> \r\n\t</value></root>") {
                        // \r\n normalized to \n
                        Err(DeError::Custom(msg)) => assert_eq!(msg, r#"invalid type: string " \n\t", expected a boolean"#),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }
            );
        };
        ($name:ident: $type:ty = $value:expr, $expected:expr) => {
            in_struct!(
                $name: $type = $value, $expected, $expected;

                /// Try to deserialize type from a root tag without data or with only spaces
                #[test]
                fn wrapped_empty() {
                    match from_str::<$type>("<root/>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, concat!(r#"invalid type: string "", expected "#, stringify!($type))),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<$type>("<root> \r\n\t</root>") {
                        // \r\n normalized to \n
                        Err(DeError::Custom(msg)) => assert_eq!(msg, concat!(r#"invalid type: string " \n\t", expected "#, stringify!($type))),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }

                #[test]
                fn field_empty() {
                    match from_str::<Field<$type>>("<root><value/></root>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, concat!(r#"invalid type: string "", expected "#, stringify!($type))),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<Field<$type>>("<root><value> \r\n\t</value></root>") {
                        // \r\n normalized to \n
                        Err(DeError::Custom(msg)) => assert_eq!(msg, concat!(r#"invalid type: string " \n\t", expected "#, stringify!($type))),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }
            );
        };
        ($name:ident: $type:ty = $value:expr, $expected:expr, $expected_with_indent:expr; $($specific_tests:item)*) => {
            mod $name {
                use super::*;

                $($specific_tests)*

                /// Try to deserialize type wrapped in a tag, optionally surround with spaces
                #[test]
                fn wrapped() {
                    let item: $type = from_str(&format!("<root>{}</root>", $value)).expect("should be Ok");
                    let expected: $type = $expected;
                    assert_eq!(item, expected);

                    let item: $type =
                        from_str(&format!("<root> \r\n\t{} \r\n\t</root>", $value)).expect("should be Ok");
                    let expected: $type = $expected_with_indent;
                    assert_eq!(item, expected);
                }

                /// Try to deserialize type wrapped in two tags
                #[test]
                fn nested() {
                    match from_str::<$type>(&format!("<root><nested>{}</nested></root>", $value)) {
                        // Expected unexpected start element `<nested>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"nested"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("nested"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<$type>(&format!(
                        "<root> \r\n\t<nested>{}</nested> \r\n\t</root>",
                        $value
                    )) {
                        // Expected unexpected start element `<nested>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"nested"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("nested"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                /// Try to deserialize type wrapped in a tag with another tag after content
                #[test]
                fn tag_after() {
                    match from_str::<$type>(&format!("<root>{}<something-else/></root>", $value)) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<$type>(&format!(
                        "<root> \r\n\t{} \r\n\t<something-else/></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                /// Try to deserialize type wrapped in a tag with another tag before content
                #[test]
                fn tag_before() {
                    match from_str::<$type>(&format!("<root><something-else/>{}</root>", $value)) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<$type>(&format!(
                        "<root><something-else/> \r\n\t{} \r\n\t</root>",
                        $value
                    )) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                /// Try to deserialize type which is a field of a struct
                #[test]
                fn field() {
                    let item: Field<$type> =
                        from_str(&format!("<root><value>{}</value></root>", $value)).expect("should be Ok");
                    assert_eq!(item, Field { value: $expected });

                    let item: Field<$type> =
                        from_str(&format!("<root><value> \r\n\t{} \r\n\t</value></root>", $value)).expect("should be Ok");
                    assert_eq!(item, Field { value: $expected_with_indent });
                }

                #[test]
                fn field_nested() {
                    match from_str::<Field<$type>>(&format!(
                        "<root><value><nested>{}</nested></value></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<nested>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"nested"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("nested"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<Field<$type>>(&format!(
                        "<root><value> \r\n\t<nested>{}</nested> \r\n\t</value></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<nested>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"nested"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("nested"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                #[test]
                fn field_tag_after() {
                    match from_str::<Field<$type>>(&format!(
                        "<root><value>{}<something-else/></value></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<Field<$type>>(&format!(
                        "<root><value> \r\n\t{} \r\n\t<something-else/></value></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                #[test]
                fn field_tag_before() {
                    match from_str::<Field<$type>>(&format!(
                        "<root><value><something-else/>{}</value></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<Field<$type>>(&format!(
                        "<root><value><something-else/> \r\n\t{} \r\n\t</value></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<something-else>`
                        Err(DeError::UnexpectedStart(tag)) => assert_eq!(tag, b"something-else"),
                        x => panic!(
                            r#"Expected `Err(UnexpectedStart("something-else"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                #[test]
                fn text_empty() {
                    match from_str::<Trivial<$type>>("<root/>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, "missing field `$text`"),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }

                    match from_str::<Trivial<$type>>("<root> \r\n\t</root>") {
                        Err(DeError::Custom(msg)) => assert_eq!(msg, "missing field `$text`"),
                        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
                    }
                }

                /// Tests deserialization from top-level tag content: `<root>...content...</root>`
                #[test]
                fn text() {
                    let item: Trivial<$type> =
                        from_str(&format!("<root>{}</root>", $value)).expect("should be Ok");
                    assert_eq!(item, Trivial { value: $expected });

                    let item: Trivial<$type> =
                        from_str(&format!("<root> \r\n\t{} \r\n\t</root>", $value)).expect("should be Ok");
                    assert_eq!(item, Trivial { value: $expected_with_indent });
                }

                #[test]
                fn text_nested() {
                    match from_str::<Trivial<$type>>(&format!(
                        "<root><nested>{}</nested></root>",
                        $value
                    )) {
                        // Expected unexpected start element `<nested>`
                        Err(DeError::Custom(reason)) => assert_eq!(reason, "missing field `$text`"),
                        x => panic!(
                            r#"Expected `Err(Custom("missing field `$text`"))`, but got `{:?}`"#,
                            x
                        ),
                    }

                    match from_str::<Trivial<$type>>(&format!(
                        "<root> \r\n\t<nested>{}</nested> \r\n\t</root>",
                        $value
                    )) {
                        // Expected unexpected start element `<nested>`
                        Err(DeError::Custom(reason)) => assert_eq!(reason, "missing field `$text`"),
                        x => panic!(
                            r#"Expected `Err(Custom("missing field `$text`"))`, but got `{:?}`"#,
                            x
                        ),
                    }
                }

                #[test]
                fn text_tag_after() {
                    // Unlike `wrapped` test, here we have a struct that is serialized to XML with
                    // an implicit field `$text` and some other field "something-else" which not interested
                    // for us in the Trivial structure. If you want the same behavior as for naked primitive,
                    // use `$value` field which would consume all data, unless a dedicated field would present
                    let item: Trivial<$type> =
                        from_str(&format!("<root>{}<something-else/></root>", $value)).expect("should be Ok");
                    assert_eq!(item, Trivial { value: $expected });

                    let item: Trivial<$type> =
                        from_str(&format!("<root> \r\n\t{} \r\n\t<something-else/></root>", $value)).expect("should be Ok");
                    assert_eq!(item, Trivial { value: $expected_with_indent });
                }

                #[test]
                fn text_tag_before() {
                    let item: Trivial<$type> =
                        from_str(&format!("<root><something-else/>{}</root>", $value)).expect("should be Ok");
                    assert_eq!(item, Trivial { value: $expected });

                    let item: Trivial<$type> =
                        from_str(&format!("<root><something-else/> \r\n\t{} \r\n\t</root>", $value)).expect("should be Ok");
                    assert_eq!(item, Trivial { value: $expected_with_indent });
                }
            }
        };
    }

#[derive(Debug, Deserialize, PartialEq)]
struct Field<T> {
    value: T,
}

/// Well-formed XML must have a single tag at the root level.
/// Any XML tag can be modeled as a struct, and content of this tag are modeled as
/// fields of this struct.
///
/// Because we want to get access to unnamed content of the tag (usually, this internal
/// XML node called `$text`) we use a rename to a special name `$text`
#[derive(Debug, Deserialize, PartialEq)]
struct Trivial<T> {
    #[serde(rename = "$text")]
    value: T,
}

/// Tests deserialization from text content in a tag
    #[rustfmt::skip] // tests formatted in a table
    mod text {
        use super::*;

        in_struct!(i8_:    i8    = "-42", -42i8);
        in_struct!(i16_:   i16   = "-4200", -4200i16);
        in_struct!(i32_:   i32   = "-42000000", -42000000i32);
        in_struct!(i64_:   i64   = "-42000000000000", -42000000000000i64);
        in_struct!(isize_: isize = "-42000000", -42000000isize);

        in_struct!(u8_:    u8    = "42", 42u8);
        in_struct!(u16_:   u16   = "4200", 4200u16);
        in_struct!(u32_:   u32   = "42000000", 42000000u32);
        in_struct!(u64_:   u64   = "42000000000000", 42000000000000u64);
        in_struct!(usize_: usize = "42000000", 42000000usize);

        in_struct!(u128_: u128 = "420000000000000000000000000000", 420000000000000000000000000000u128);
        in_struct!(i128_: i128 = "-420000000000000000000000000000", -420000000000000000000000000000i128);

        in_struct!(f32_: f32 = "4.2", 4.2f32);
        in_struct!(f64_: f64 = "4.2", 4.2f64);

        in_struct!(false_: bool = "false", false);
        in_struct!(true_: bool = "true", true);
        in_struct!(char_: char = "r", 'r');

        in_struct!(string: String = "escaped&#x20;string", "escaped string");

        /// XML does not able to store binary data
        #[test]
        fn byte_buf() {
            match from_str::<Trivial<ByteBuf>>("<root>escaped&#x20;byte_buf</root>") {
                Err(DeError::Custom(msg)) => {
                    assert_eq!(msg, "invalid type: string \"escaped byte_buf\", expected byte data")
                }
                x => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"escaped byte_buf\", expected byte data"))`, but got `{:?}`"#,
                    x
                ),
            }
        }

        /// XML does not able to store binary data
        #[test]
        fn bytes() {
            match from_str::<Trivial<Bytes>>("<root>escaped&#x20;byte_buf</root>") {
                Err(DeError::Custom(msg)) => {
                    assert_eq!(msg, "invalid type: string \"escaped byte_buf\", expected borrowed bytes")
                }
                x => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"escaped byte_buf\", expected borrowed bytes"))`, but got `{:?}`"#,
                    x
                ),
            }
        }
    }

/// Tests deserialization from CDATA content in a tag.
    /// CDATA handling similar to text handling except that strings does not unescapes
    #[rustfmt::skip] // tests formatted in a table
    mod cdata {
        use super::*;

        in_struct!(i8_:    i8    = "<![CDATA[-42]]>", -42i8);
        in_struct!(i16_:   i16   = "<![CDATA[-4200]]>", -4200i16);
        in_struct!(i32_:   i32   = "<![CDATA[-42000000]]>", -42000000i32);
        in_struct!(i64_:   i64   = "<![CDATA[-42000000000000]]>", -42000000000000i64);
        in_struct!(isize_: isize = "<![CDATA[-42000000]]>", -42000000isize);

        in_struct!(u8_:    u8    = "<![CDATA[42]]>", 42u8);
        in_struct!(u16_:   u16   = "<![CDATA[4200]]>", 4200u16);
        in_struct!(u32_:   u32   = "<![CDATA[42000000]]>", 42000000u32);
        in_struct!(u64_:   u64   = "<![CDATA[42000000000000]]>", 42000000000000u64);
        in_struct!(usize_: usize = "<![CDATA[42000000]]>", 42000000usize);

        in_struct!(u128_: u128 = "<![CDATA[420000000000000000000000000000]]>", 420000000000000000000000000000u128);
        in_struct!(i128_: i128 = "<![CDATA[-420000000000000000000000000000]]>", -420000000000000000000000000000i128);

        in_struct!(f32_: f32 = "<![CDATA[4.2]]>", 4.2f32);
        in_struct!(f64_: f64 = "<![CDATA[4.2]]>", 4.2f64);

        in_struct!(false_: bool = "<![CDATA[false]]>", false);
        in_struct!(true_: bool = "<![CDATA[true]]>", true);
        in_struct!(char_: char = "<![CDATA[r]]>", 'r');

        // Escape sequences does not processed inside CDATA section
        in_struct!(string: String = "<![CDATA[escaped&#x20;string]]>", "escaped&#x20;string");

        /// XML does not able to store binary data
        #[test]
        fn byte_buf() {
            match from_str::<Trivial<ByteBuf>>("<root><![CDATA[escaped&#x20;byte_buf]]></root>") {
                Err(DeError::Custom(msg)) => {
                    assert_eq!(msg, "invalid type: string \"escaped&#x20;byte_buf\", expected byte data")
                }
                x => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"escaped&#x20;byte_buf\", expected byte data"))`, but got `{:?}`"#,
                    x
                ),
            }
        }

        /// XML does not able to store binary data
        #[test]
        fn bytes() {
            match from_str::<Trivial<Bytes>>("<root><![CDATA[escaped&#x20;byte_buf]]></root>") {
                Err(DeError::Custom(msg)) => {
                    assert_eq!(msg, "invalid type: string \"escaped&#x20;byte_buf\", expected borrowed bytes")
                }
                x => panic!(
                    r#"Expected `Err(Custom("invalid type: string \"escaped&#x20;byte_buf\", expected borrowed bytes"))`, but got `{:?}`"#,
                    x
                ),
            }
        }
    }
