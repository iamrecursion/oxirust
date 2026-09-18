//! Unit tests for [`ElementSerializer`](super::ElementSerializer).

use super::*;
use crate::se::content::tests::*;
use crate::se::{Indent, QuoteLevel, TextFormat};
use crate::utils::Bytes;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize, PartialEq)]
struct OptionalElements {
    a: Option<&'static str>,

    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<&'static str>,
}
#[derive(Debug, Serialize, PartialEq)]
struct OptionalAttributes {
    #[serde(rename = "@a")]
    a: Option<&'static str>,

    #[serde(rename = "@b")]
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<&'static str>,
}

mod without_indent {
    use super::*;
    use crate::se::content::tests::Struct;

    /// Checks that given `$data` successfully serialized as `$expected`
    macro_rules! serialize_as {
        ($name:ident: $data:expr => $expected:expr) => {
            #[test]
            fn $name() {
                let mut buffer = String::new();
                let ser = ElementSerializer {
                    ser: ContentSerializer {
                        writer: &mut buffer,
                        level: QuoteLevel::Full,
                        indent: Indent::None,
                        write_indent: false,
                        text_format: TextFormat::Text,
                        allow_primitive: true,
                        empty_element_handling: EmptyElementHandling::SelfClosed,
                    },
                    key: XmlName("root"),
                };

                let result = $data.serialize(ser).expect("serialization should succeed");
                assert_eq!(buffer, $expected);
                assert_eq!(result, WriteResult::Element);
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
                let ser = ElementSerializer {
                    ser: ContentSerializer {
                        writer: &mut buffer,
                        level: QuoteLevel::Full,
                        indent: Indent::None,
                        write_indent: false,
                        text_format: TextFormat::Text,
                        allow_primitive: true,
                        empty_element_handling: EmptyElementHandling::SelfClosed,
                    },
                    key: XmlName("root"),
                };

                match $data.serialize(ser).expect_err("serialization should fail") {
                    SeError::$kind(e) => assert_eq!(e, $reason),
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

    serialize_as!(char_non_escaped: 'h' => "<root>h</root>");
    serialize_as!(char_lt:   '<' => "<root>&lt;</root>");
    serialize_as!(char_gt:   '>' => "<root>&gt;</root>");
    serialize_as!(char_amp:  '&' => "<root>&amp;</root>");
    serialize_as!(char_apos: '\'' => "<root>&apos;</root>");
    serialize_as!(char_quot: '"' => "<root>&quot;</root>");

    serialize_as!(str_non_escaped: "non-escaped string" => "<root>non-escaped string</root>");
    serialize_as!(str_escaped: "<\"escaped & string'>" => "<root>&lt;&quot;escaped &amp; string&apos;&gt;</root>");

    err!(bytes: Bytes(b"<\"escaped & bytes'>") => Unsupported("`serialize_bytes` not supported yet"));

    serialize_as!(option_none: Option::<&str>::None => "<root/>");
    serialize_as!(option_some: Some("non-escaped string") => "<root>non-escaped string</root>");
    serialize_as!(option_some_empty_str: Some("") => "<root/>");

    serialize_as!(unit: () => "<root/>");
    serialize_as!(unit_struct: Unit => "<root/>");
    serialize_as!(unit_struct_escaped: UnitEscaped => "<root/>");

    serialize_as!(enum_unit: Enum::Unit => "<root>Unit</root>");
    serialize_as!(enum_unit_escaped: Enum::UnitEscaped => "<root>&lt;&quot;&amp;&apos;&gt;</root>");

    serialize_as!(newtype: Newtype(42) => "<root>42</root>");
    err!(enum_newtype: Enum::Newtype(42)
        => Unsupported("cannot serialize enum newtype variant `Enum::Newtype`"));

    serialize_as!(seq: vec![1, 2, 3]
        => "<root>1</root>\
            <root>2</root>\
            <root>3</root>");
    serialize_as!(seq_empty: Vec::<usize>::new() => "");
    serialize_as!(tuple: ("<\"&'>", "with\t\n\r spaces", 3usize)
        => "<root>&lt;&quot;&amp;&apos;&gt;</root>\
            <root>with\t\n\r spaces</root>\
            <root>3</root>");
    serialize_as!(tuple_struct: Tuple("first", 42)
        => "<root>first</root>\
            <root>42</root>");
    err!(enum_tuple: Enum::Tuple("first", 42)
        => Unsupported("cannot serialize enum tuple variant `Enum::Tuple`"));

    serialize_as!(map: BTreeMap::from([("_1", 2), ("_3", 4)])
        => "<root>\
                <_1>2</_1>\
                <_3>4</_3>\
            </root>");
    serialize_as!(struct_: Struct { key: "answer", val: (42, 42) }
        => "<root>\
                <key>answer</key>\
                <val>42</val>\
                <val>42</val>\
            </root>");
    err!(enum_struct: Enum::Struct { key: "answer", val: (42, 42) }
        => Unsupported("cannot serialize enum struct variant `Enum::Struct`"));

    /// Special field name `$text` should be serialized as text content.
    /// Sequences serialized as an `xs:list` content
    mod text_field {
        use super::*;

        /// `$text` key in a map
        mod map {
            use super::*;

            macro_rules! text {
                ($name:ident: $data:expr) => {
                    serialize_as!($name:
                        BTreeMap::from([("$text", $data)])
                        => "<root/>");
                };
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        BTreeMap::from([("$text", $data)])
                        => concat!("<root>", $expected,"</root>"));
                };
            }

            text!(false_: false => "false");
            text!(true_:  true  => "true");

            text!(i8_:    -42i8                => "-42");
            text!(i16_:   -4200i16             => "-4200");
            text!(i32_:   -42000000i32         => "-42000000");
            text!(i64_:   -42000000000000i64   => "-42000000000000");
            text!(isize_: -42000000isize       => "-42000000");

            text!(u8_:    42u8                => "42");
            text!(u16_:   4200u16             => "4200");
            text!(u32_:   42000000u32         => "42000000");
            text!(u64_:   42000000000000u64   => "42000000000000");
            text!(usize_: 42000000usize       => "42000000");

            text!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            text!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            text!(f32_: 4.2f32 => "4.2");
            text!(f64_: 4.2f64 => "4.2");

            text!(char_non_escaped: 'h' => "h");
            text!(char_lt:   '<' => "&lt;");
            text!(char_gt:   '>' => "&gt;");
            text!(char_amp:  '&' => "&amp;");
            text!(char_apos: '\'' => "&apos;");
            text!(char_quot: '"' => "&quot;");
            text!(char_space: ' ' => " ");

            text!(str_non_escaped: "non-escaped string" => "non-escaped string");
            text!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                Text {
                    before: "answer",
                    content: Bytes(b"<\"escaped & bytes'>"),
                    after: "answer",
                }
                => Unsupported("`serialize_bytes` not supported yet"));

            text!(option_none: Option::<&str>::None);
            text!(option_some: Some("non-escaped string") => "non-escaped string");
            text!(option_some_empty_str: Some(""));

            text!(unit: ());
            text!(unit_struct: Unit);
            text!(unit_struct_escaped: UnitEscaped);

            text!(enum_unit: Enum::Unit => "Unit");
            text!(enum_unit_escaped: Enum::UnitEscaped => "&lt;&quot;&amp;&apos;&gt;");

            text!(newtype: Newtype(42) => "42");
            // We have no space where name of a variant can be stored
            err!(enum_newtype:
                Text {
                    before: "answer",
                    content: Enum::Newtype(42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum newtype variant `Enum::Newtype` as text content value"));

            // Sequences are serialized separated by spaces, all spaces inside are escaped
            text!(seq: vec![1, 2, 3] => "1 2 3");
            text!(seq_empty: Vec::<usize>::new());
            text!(tuple: ("<\"&'>", "with\t\n\r spaces", 3usize)
                => "&lt;&quot;&amp;&apos;&gt; \
                    with&#9;&#10;&#13;&#32;spaces \
                    3");
            text!(tuple_struct: Tuple("first", 42) => "first 42");
            // We have no space where name of a variant can be stored
            err!(enum_tuple:
                Text {
                    before: "answer",
                    content: Enum::Tuple("first", 42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum tuple variant `Enum::Tuple` as text content value"));

            // Complex types cannot be serialized in `$text` field
            err!(map:
                Text {
                    before: "answer",
                    content: BTreeMap::from([("_1", 2), ("_3", 4)]),
                    after: "answer",
                }
                => Unsupported("cannot serialize map as text content value"));
            err!(struct_:
                Text {
                    before: "answer",
                    content: Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize struct `Struct` as text content value"));
            err!(enum_struct:
                Text {
                    before: "answer",
                    content: Enum::Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize enum struct variant `Enum::Struct` as text content value"));
        }

        /// `$text` field inside a struct
        mod struct_ {
            use super::*;

            macro_rules! text {
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        Text {
                            before: "answer",
                            content: $data,
                            after: "answer",
                        }
                        => concat!(
                            "<root><before>answer</before>",
                            $expected,
                            "<after>answer</after></root>",
                        ));
                };
            }

            text!(false_: false => "false");
            text!(true_:  true  => "true");

            text!(i8_:    -42i8                => "-42");
            text!(i16_:   -4200i16             => "-4200");
            text!(i32_:   -42000000i32         => "-42000000");
            text!(i64_:   -42000000000000i64   => "-42000000000000");
            text!(isize_: -42000000isize       => "-42000000");

            text!(u8_:    42u8                => "42");
            text!(u16_:   4200u16             => "4200");
            text!(u32_:   42000000u32         => "42000000");
            text!(u64_:   42000000000000u64   => "42000000000000");
            text!(usize_: 42000000usize       => "42000000");

            text!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            text!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            text!(f32_: 4.2f32 => "4.2");
            text!(f64_: 4.2f64 => "4.2");

            text!(char_non_escaped: 'h' => "h");
            text!(char_lt:   '<' => "&lt;");
            text!(char_gt:   '>' => "&gt;");
            text!(char_amp:  '&' => "&amp;");
            text!(char_apos: '\'' => "&apos;");
            text!(char_quot: '"' => "&quot;");
            text!(char_space: ' ' => " ");

            text!(str_non_escaped: "non-escaped string" => "non-escaped string");
            text!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                Text {
                    before: "answer",
                    content: Bytes(b"<\"escaped & bytes'>"),
                    after: "answer",
                }
                => Unsupported("`serialize_bytes` not supported yet"));

            text!(option_none: Option::<&str>::None => "");
            text!(option_some: Some("non-escaped string") => "non-escaped string");
            text!(option_some_empty_str: Some("") => "");

            text!(unit: () => "");
            text!(unit_struct: Unit => "");
            text!(unit_struct_escaped: UnitEscaped => "");

            text!(enum_unit: Enum::Unit => "Unit");
            text!(enum_unit_escaped: Enum::UnitEscaped => "&lt;&quot;&amp;&apos;&gt;");

            text!(newtype: Newtype(42) => "42");
            // We have no space where name of a variant can be stored
            err!(enum_newtype:
                Text {
                    before: "answer",
                    content: Enum::Newtype(42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum newtype variant `Enum::Newtype` as text content value"));

            // Sequences are serialized separated by spaces, all spaces inside are escaped
            text!(seq: vec![1, 2, 3] => "1 2 3");
            text!(seq_empty: Vec::<usize>::new() => "");
            text!(tuple: ("<\"&'>", "with\t\n\r spaces", 3usize)
                => "&lt;&quot;&amp;&apos;&gt; \
                    with&#9;&#10;&#13;&#32;spaces \
                    3");
            text!(tuple_struct: Tuple("first", 42) => "first 42");
            // We have no space where name of a variant can be stored
            err!(enum_tuple:
                Text {
                    before: "answer",
                    content: Enum::Tuple("first", 42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum tuple variant `Enum::Tuple` as text content value"));

            // Complex types cannot be serialized in `$text` field
            err!(map:
                Text {
                    before: "answer",
                    content: BTreeMap::from([("_1", 2), ("_3", 4)]),
                    after: "answer",
                }
                => Unsupported("cannot serialize map as text content value"));
            err!(struct_:
                Text {
                    before: "answer",
                    content: Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize struct `Struct` as text content value"));
            err!(enum_struct:
                Text {
                    before: "answer",
                    content: Enum::Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize enum struct variant `Enum::Struct` as text content value"));
        }
    }

    /// Special field name `$value` should be serialized using name, provided
    /// by the type of value instead of a key. Sequences serialized as a list
    /// of tags with that name (each element can have their own name)
    mod value_field {
        use super::*;

        /// `$value` key in a map
        mod map {
            use super::*;

            macro_rules! value {
                ($name:ident: $data:expr) => {
                    serialize_as!($name:
                        BTreeMap::from([("$value", $data)])
                        => "<root/>");
                };
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        BTreeMap::from([("$value", $data)])
                        => concat!("<root>", $expected,"</root>"));
                };
            }

            value!(false_: false => "false");
            value!(true_:  true  => "true");

            value!(i8_:    -42i8                => "-42");
            value!(i16_:   -4200i16             => "-4200");
            value!(i32_:   -42000000i32         => "-42000000");
            value!(i64_:   -42000000000000i64   => "-42000000000000");
            value!(isize_: -42000000isize       => "-42000000");

            value!(u8_:    42u8                => "42");
            value!(u16_:   4200u16             => "4200");
            value!(u32_:   42000000u32         => "42000000");
            value!(u64_:   42000000000000u64   => "42000000000000");
            value!(usize_: 42000000usize       => "42000000");

            value!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            value!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            value!(f32_: 4.2f32 => "4.2");
            value!(f64_: 4.2f64 => "4.2");

            value!(char_non_escaped: 'h' => "h");
            value!(char_lt:   '<' => "&lt;");
            value!(char_gt:   '>' => "&gt;");
            value!(char_amp:  '&' => "&amp;");
            value!(char_apos: '\'' => "&apos;");
            value!(char_quot: '"' => "&quot;");
            value!(char_space: ' ' => " ");

            value!(str_non_escaped: "non-escaped string" => "non-escaped string");
            value!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                BTreeMap::from([("$value", Bytes(b"<\"escaped & bytes'>"))])
                => Unsupported("`serialize_bytes` not supported yet"));

            value!(option_none: Option::<&str>::None);
            value!(option_some: Some("non-escaped string") => "non-escaped string");
            value!(option_some_empty_str: Some(""));

            value!(unit: ());
            value!(unit_struct: Unit);
            value!(unit_struct_escaped: UnitEscaped);

            value!(enum_unit: Enum::Unit => "<Unit/>");
            err!(enum_unit_escaped:
                BTreeMap::from([("$value", Enum::UnitEscaped)])
                => Unsupported("character `<` is not allowed at the start of an XML name `<\"&'>`"));

            value!(newtype: Newtype(42) => "42");
            value!(enum_newtype: Enum::Newtype(42) => "<Newtype>42</Newtype>");

            // Note that sequences of primitives serialized without delimiters!
            err!(seq:
                BTreeMap::from([("$value", vec![1, 2, 3])])
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(seq_empty: Vec::<usize>::new());
            err!(tuple:
                BTreeMap::from([("$value", ("<\"&'>", "with\t\n\r spaces", 3usize))])
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            err!(tuple_struct:
                BTreeMap::from([("$value", Tuple("first", 42))])
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(enum_tuple: Enum::Tuple("first", 42)
                => "<Tuple>first</Tuple>\
                    <Tuple>42</Tuple>");

            // We cannot wrap map in any container and should not
            // flatten it, so it is impossible to serialize maps
            err!(map:
                BTreeMap::from([("$value", BTreeMap::from([("_1", 2), ("_3", 4)]))])
                => Unsupported("serialization of map types is not supported in `$value` field"));
            value!(struct_:
                Struct { key: "answer", val: (42, 42) }
                => "<Struct>\
                        <key>answer</key>\
                        <val>42</val>\
                        <val>42</val>\
                    </Struct>");
            value!(enum_struct:
                Enum::Struct { key: "answer", val: (42, 42) }
                => "<Struct>\
                        <key>answer</key>\
                        <val>42</val>\
                        <val>42</val>\
                    </Struct>");
        }

        /// `$value` field inside a struct
        mod struct_ {
            use super::*;

            macro_rules! value {
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        Value {
                            before: "answer",
                            content: $data,
                            after: "answer",
                        }
                        => concat!(
                            "<root><before>answer</before>",
                            $expected,
                            "<after>answer</after></root>",
                        ));
                };
            }

            value!(false_: false => "false");
            value!(true_:  true  => "true");

            value!(i8_:    -42i8                => "-42");
            value!(i16_:   -4200i16             => "-4200");
            value!(i32_:   -42000000i32         => "-42000000");
            value!(i64_:   -42000000000000i64   => "-42000000000000");
            value!(isize_: -42000000isize       => "-42000000");

            value!(u8_:    42u8                => "42");
            value!(u16_:   4200u16             => "4200");
            value!(u32_:   42000000u32         => "42000000");
            value!(u64_:   42000000000000u64   => "42000000000000");
            value!(usize_: 42000000usize       => "42000000");

            value!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            value!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            value!(f32_: 4.2f32 => "4.2");
            value!(f64_: 4.2f64 => "4.2");

            value!(char_non_escaped: 'h' => "h");
            value!(char_lt:   '<' => "&lt;");
            value!(char_gt:   '>' => "&gt;");
            value!(char_amp:  '&' => "&amp;");
            value!(char_apos: '\'' => "&apos;");
            value!(char_quot: '"' => "&quot;");
            value!(char_space: ' ' => " ");

            value!(str_non_escaped: "non-escaped string" => "non-escaped string");
            value!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                Value {
                    before: "answer",
                    content: Bytes(b"<\"escaped & bytes'>"),
                    after: "answer",
                }
                => Unsupported("`serialize_bytes` not supported yet"));

            value!(option_none: Option::<&str>::None => "");
            value!(option_some: Some("non-escaped string") => "non-escaped string");
            value!(option_some_empty_str: Some("") => "");

            value!(unit: () => "");
            value!(unit_struct: Unit => "");
            value!(unit_struct_escaped: UnitEscaped => "");

            value!(enum_unit: Enum::Unit => "<Unit/>");
            err!(enum_unit_escaped:
                Value {
                    before: "answer",
                    content: Enum::UnitEscaped,
                    after: "answer",
                }
                => Unsupported("character `<` is not allowed at the start of an XML name `<\"&'>`"));

            value!(newtype: Newtype(42) => "42");
            value!(enum_newtype: Enum::Newtype(42) => "<Newtype>42</Newtype>");

            // Note that sequences of primitives serialized without delimiters!
            err!(seq:
                Value {
                    before: "answer",
                    content: vec![1, 2, 3],
                    after: "answer",
                }
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(seq_empty: Vec::<usize>::new() => "");
            err!(tuple:
                Value {
                    before: "answer",
                    content: ("<\"&'>", "with\t\n\r spaces", 3usize),
                    after: "answer",
                }
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            err!(tuple_struct:
                Value {
                    before: "answer",
                    content: Tuple("first", 42),
                    after: "answer",
                }
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(enum_tuple: Enum::Tuple("first", 42)
                => "<Tuple>first</Tuple>\
                    <Tuple>42</Tuple>");

            // We cannot wrap map in any container and should not
            // flatten it, so it is impossible to serialize maps
            err!(map:
                Value {
                    before: "answer",
                    content: BTreeMap::from([("_1", 2), ("_3", 4)]),
                    after: "answer",
                }
                => Unsupported("serialization of map types is not supported in `$value` field"));
            value!(struct_:
                Struct { key: "answer", val: (42, 42) }
                => "<Struct>\
                        <key>answer</key>\
                        <val>42</val>\
                        <val>42</val>\
                    </Struct>");
            value!(enum_struct:
                Enum::Struct { key: "answer", val: (42, 42) }
                => "<Struct>\
                        <key>answer</key>\
                        <val>42</val>\
                        <val>42</val>\
                    </Struct>");
        }
    }

    mod attributes {
        use super::*;

        serialize_as!(map_attr: BTreeMap::from([("@key1", 1), ("@key2", 2)])
            => r#"<root key1="1" key2="2"/>"#);
        serialize_as!(map_mixed: BTreeMap::from([("@key1", 1), ("key2", 2)])
            => r#"<root key1="1"><key2>2</key2></root>"#);

        serialize_as!(struct_: Attributes { key: "answer", val: (42, 42) }
            => r#"<root key="answer" val="42 42"/>"#);
        serialize_as!(struct_before: AttributesBefore { key: "answer", val: 42 }
            => r#"<root key="answer"><val>42</val></root>"#);
        serialize_as!(struct_after: AttributesAfter { key: "answer", val: 42 }
            => r#"<root val="42"><key>answer</key></root>"#);

        err!(enum_: Enum::Attributes { key: "answer", val: (42, 42) }
            => Unsupported("cannot serialize enum struct variant `Enum::Attributes`"));

        /// Test for https://github.com/tafia/quick-xml/issues/252
        mod optional {
            use super::*;

            serialize_as!(none:
                OptionalAttributes { a: None, b: None }
                => r#"<root a=""/>"#);
            serialize_as!(some_empty_str:
                OptionalAttributes {
                    a: Some(""),
                    b: Some(""),
                }
                => r#"<root a="" b=""/>"#);
            serialize_as!(some_non_empty:
                OptionalAttributes {
                    a: Some("1"),
                    b: Some("2"),
                }
                => r#"<root a="1" b="2"/>"#);
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/252
    mod optional {
        use super::*;

        serialize_as!(none:
            OptionalElements { a: None, b: None }
            => "<root>\
                    <a/>\
                </root>");
        serialize_as!(some_empty_str:
            OptionalElements {
                a: Some(""),
                b: Some(""),
            }
            => "<root>\
                    <a/>\
                    <b/>\
                </root>");
        serialize_as!(some_non_empty:
            OptionalElements {
                a: Some("1"),
                b: Some("2"),
            }
            => "<root>\
                    <a>1</a>\
                    <b>2</b>\
                </root>");
    }
}

mod with_indent {
    use super::*;
    use crate::se::content::tests::Struct;
    use crate::writer::Indentation;

    /// Checks that given `$data` successfully serialized as `$expected`.
    /// Writes `$data` using [`ElementSerializer`] with indent of two spaces.
    macro_rules! serialize_as {
        ($name:ident: $data:expr => $expected:expr) => {
            #[test]
            fn $name() {
                let mut buffer = String::new();
                let ser = ElementSerializer {
                    ser: ContentSerializer {
                        writer: &mut buffer,
                        level: QuoteLevel::Full,
                        indent: Indent::Owned(Indentation::new(b' ', 2)),
                        write_indent: false,
                        text_format: TextFormat::Text,
                        allow_primitive: true,
                        empty_element_handling: EmptyElementHandling::SelfClosed,
                    },
                    key: XmlName("root"),
                };

                let result = $data.serialize(ser).expect("serialization should succeed");
                assert_eq!(buffer, $expected);
                assert_eq!(result, WriteResult::Element);
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
                let ser = ElementSerializer {
                    ser: ContentSerializer {
                        writer: &mut buffer,
                        level: QuoteLevel::Full,
                        indent: Indent::Owned(Indentation::new(b' ', 2)),
                        write_indent: false,
                        text_format: TextFormat::Text,
                        allow_primitive: true,
                        empty_element_handling: EmptyElementHandling::SelfClosed,
                    },
                    key: XmlName("root"),
                };

                match $data.serialize(ser).expect_err("serialization should fail") {
                    SeError::$kind(e) => assert_eq!(e, $reason),
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

    serialize_as!(char_non_escaped: 'h' => "<root>h</root>");
    serialize_as!(char_lt:   '<' => "<root>&lt;</root>");
    serialize_as!(char_gt:   '>' => "<root>&gt;</root>");
    serialize_as!(char_amp:  '&' => "<root>&amp;</root>");
    serialize_as!(char_apos: '\'' => "<root>&apos;</root>");
    serialize_as!(char_quot: '"' => "<root>&quot;</root>");
    serialize_as!(char_space: ' ' => "<root> </root>");

    serialize_as!(str_non_escaped: "non-escaped string" => "<root>non-escaped string</root>");
    serialize_as!(str_escaped: "<\"escaped & string'>" => "<root>&lt;&quot;escaped &amp; string&apos;&gt;</root>");

    err!(bytes: Bytes(b"<\"escaped & bytes'>") => Unsupported("`serialize_bytes` not supported yet"));

    serialize_as!(option_none: Option::<&str>::None => "<root/>");
    serialize_as!(option_some: Some("non-escaped string") => "<root>non-escaped string</root>");
    serialize_as!(option_some_empty: Some("") => "<root/>");

    serialize_as!(unit: () => "<root/>");
    serialize_as!(unit_struct: Unit => "<root/>");
    serialize_as!(unit_struct_escaped: UnitEscaped => "<root/>");

    serialize_as!(enum_unit: Enum::Unit => "<root>Unit</root>");
    serialize_as!(enum_unit_escaped: Enum::UnitEscaped => "<root>&lt;&quot;&amp;&apos;&gt;</root>");

    serialize_as!(newtype: Newtype(42) => "<root>42</root>");
    err!(enum_newtype: Enum::Newtype(42)
        => Unsupported("cannot serialize enum newtype variant `Enum::Newtype`"));

    serialize_as!(seq: vec![1, 2, 3]
        => "<root>1</root>\n\
            <root>2</root>\n\
            <root>3</root>");
    serialize_as!(seq_empty: Vec::<usize>::new() => "");
    serialize_as!(tuple: ("<\"&'>", "with\t\n\r spaces", 3usize)
        => "<root>&lt;&quot;&amp;&apos;&gt;</root>\n\
            <root>with\t\n\r spaces</root>\n\
            <root>3</root>");
    serialize_as!(tuple_struct: Tuple("first", 42)
        => "<root>first</root>\n\
            <root>42</root>");
    err!(enum_tuple: Enum::Tuple("first", 42)
        => Unsupported("cannot serialize enum tuple variant `Enum::Tuple`"));

    serialize_as!(map: BTreeMap::from([("_1", 2), ("_3", 4)])
        => "<root>\n  \
                <_1>2</_1>\n  \
                <_3>4</_3>\n\
            </root>");
    serialize_as!(struct_: Struct { key: "answer", val: (42, 42) }
        => "<root>\n  \
                <key>answer</key>\n  \
                <val>42</val>\n  \
                <val>42</val>\n\
            </root>");
    err!(enum_struct: Enum::Struct { key: "answer", val: (42, 42) }
        => Unsupported("cannot serialize enum struct variant `Enum::Struct`"));

    /// Special field name `$text` should be serialized as text content.
    /// Sequences serialized as an `xs:list` content
    mod text_field {
        use super::*;

        /// `$text` key in a map
        mod map {
            use super::*;

            macro_rules! text {
                ($name:ident: $data:expr) => {
                    serialize_as!($name:
                        // Serialization started from ElementSerializer::serialize_map
                        BTreeMap::from([("$text", $data)])
                        => "<root/>");
                };
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        // Serialization started from ElementSerializer::serialize_map
                        BTreeMap::from([("$text", $data)])
                        => concat!("<root>", $expected,"</root>"));
                };
            }

            text!(false_: false => "false");
            text!(true_:  true  => "true");

            text!(i8_:    -42i8                => "-42");
            text!(i16_:   -4200i16             => "-4200");
            text!(i32_:   -42000000i32         => "-42000000");
            text!(i64_:   -42000000000000i64   => "-42000000000000");
            text!(isize_: -42000000isize       => "-42000000");

            text!(u8_:    42u8                => "42");
            text!(u16_:   4200u16             => "4200");
            text!(u32_:   42000000u32         => "42000000");
            text!(u64_:   42000000000000u64   => "42000000000000");
            text!(usize_: 42000000usize       => "42000000");

            text!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            text!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            text!(f32_: 4.2f32 => "4.2");
            text!(f64_: 4.2f64 => "4.2");

            text!(char_non_escaped: 'h' => "h");
            text!(char_lt:   '<' => "&lt;");
            text!(char_gt:   '>' => "&gt;");
            text!(char_amp:  '&' => "&amp;");
            text!(char_apos: '\'' => "&apos;");
            text!(char_quot: '"' => "&quot;");
            text!(char_space: ' ' => " ");

            text!(str_non_escaped: "non-escaped string" => "non-escaped string");
            text!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                Text {
                    before: "answer",
                    content: Bytes(b"<\"escaped & bytes'>"),
                    after: "answer",
                }
                => Unsupported("`serialize_bytes` not supported yet"));

            text!(option_none: Option::<&str>::None);
            text!(option_some: Some("non-escaped string") => "non-escaped string");
            text!(option_some_empty_str: Some(""));

            text!(unit: ());
            text!(unit_struct: Unit);
            text!(unit_struct_escaped: UnitEscaped);

            text!(enum_unit: Enum::Unit => "Unit");
            text!(enum_unit_escaped: Enum::UnitEscaped => "&lt;&quot;&amp;&apos;&gt;");

            text!(newtype: Newtype(42) => "42");
            // We have no space where name of a variant can be stored
            err!(enum_newtype:
                Text {
                    before: "answer",
                    content: Enum::Newtype(42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum newtype variant `Enum::Newtype` as text content value"));

            // Sequences are serialized separated by spaces, all spaces inside are escaped
            text!(seq: vec![1, 2, 3] => "1 2 3");
            text!(seq_empty: Vec::<usize>::new());
            text!(tuple: ("<\"&'>", "with\t\n\r spaces", 3usize)
                => "&lt;&quot;&amp;&apos;&gt; \
                    with&#9;&#10;&#13;&#32;spaces \
                    3");
            text!(tuple_struct: Tuple("first", 42) => "first 42");
            // We have no space where name of a variant can be stored
            err!(enum_tuple:
                Text {
                    before: "answer",
                    content: Enum::Tuple("first", 42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum tuple variant `Enum::Tuple` as text content value"));

            // Complex types cannot be serialized in `$text` field
            err!(map:
                Text {
                    before: "answer",
                    content: BTreeMap::from([("_1", 2), ("_3", 4)]),
                    after: "answer",
                }
                => Unsupported("cannot serialize map as text content value"));
            err!(struct_:
                Text {
                    before: "answer",
                    content: Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize struct `Struct` as text content value"));
            err!(enum_struct:
                Text {
                    before: "answer",
                    content: Enum::Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize enum struct variant `Enum::Struct` as text content value"));
        }

        /// `$text` field inside a struct
        mod struct_ {
            use super::*;

            macro_rules! text {
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        // Serialization started from ElementSerializer::serialize_struct
                        Text {
                            before: "answer",
                            content: $data,
                            after: "answer",
                        }
                        => concat!(
                            "<root>\n  <before>answer</before>",
                            $expected,
                            "<after>answer</after>\n</root>",
                        ));
                };
            }

            text!(false_: false => "false");
            text!(true_:  true  => "true");

            text!(i8_:    -42i8                => "-42");
            text!(i16_:   -4200i16             => "-4200");
            text!(i32_:   -42000000i32         => "-42000000");
            text!(i64_:   -42000000000000i64   => "-42000000000000");
            text!(isize_: -42000000isize       => "-42000000");

            text!(u8_:    42u8                => "42");
            text!(u16_:   4200u16             => "4200");
            text!(u32_:   42000000u32         => "42000000");
            text!(u64_:   42000000000000u64   => "42000000000000");
            text!(usize_: 42000000usize       => "42000000");

            text!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            text!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            text!(f32_: 4.2f32 => "4.2");
            text!(f64_: 4.2f64 => "4.2");

            text!(char_non_escaped: 'h' => "h");
            text!(char_lt:   '<' => "&lt;");
            text!(char_gt:   '>' => "&gt;");
            text!(char_amp:  '&' => "&amp;");
            text!(char_apos: '\'' => "&apos;");
            text!(char_quot: '"' => "&quot;");
            text!(char_space: ' ' => " ");

            text!(str_non_escaped: "non-escaped string" => "non-escaped string");
            text!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                Text {
                    before: "answer",
                    content: Bytes(b"<\"escaped & bytes'>"),
                    after: "answer",
                }
                => Unsupported("`serialize_bytes` not supported yet"));

            text!(option_none: Option::<&str>::None => "");
            text!(option_some: Some("non-escaped string") => "non-escaped string");
            text!(option_some_empty_str: Some("") => "");

            text!(unit: () => "");
            text!(unit_struct: Unit => "");
            text!(unit_struct_escaped: UnitEscaped => "");

            text!(enum_unit: Enum::Unit => "Unit");
            text!(enum_unit_escaped: Enum::UnitEscaped => "&lt;&quot;&amp;&apos;&gt;");

            text!(newtype: Newtype(42) => "42");
            // We have no space where name of a variant can be stored
            err!(enum_newtype:
                Text {
                    before: "answer",
                    content: Enum::Newtype(42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum newtype variant `Enum::Newtype` as text content value"));

            // Sequences are serialized separated by spaces, all spaces inside are escaped
            text!(seq: vec![1, 2, 3] => "1 2 3");
            text!(seq_empty: Vec::<usize>::new() => "");
            text!(tuple: ("<\"&'>", "with\t\n\r spaces", 3usize)
                => "&lt;&quot;&amp;&apos;&gt; \
                    with&#9;&#10;&#13;&#32;spaces \
                    3");
            text!(tuple_struct: Tuple("first", 42) => "first 42");
            // We have no space where name of a variant can be stored
            err!(enum_tuple:
                Text {
                    before: "answer",
                    content: Enum::Tuple("first", 42),
                    after: "answer",
                }
                => Unsupported("cannot serialize enum tuple variant `Enum::Tuple` as text content value"));

            // Complex types cannot be serialized in `$text` field
            err!(map:
                Text {
                    before: "answer",
                    content: BTreeMap::from([("_1", 2), ("_3", 4)]),
                    after: "answer",
                }
                => Unsupported("cannot serialize map as text content value"));
            err!(struct_:
                Text {
                    before: "answer",
                    content: Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize struct `Struct` as text content value"));
            err!(enum_struct:
                Text {
                    before: "answer",
                    content: Enum::Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => Unsupported("cannot serialize enum struct variant `Enum::Struct` as text content value"));
        }
    }

    /// Special field name `$value` should be serialized using name, provided
    /// by the type of value instead of a key. Sequences serialized as a list
    /// of tags with that name (each element can have their own name)
    mod value_field {
        use super::*;

        /// `$value` key in a map
        mod map {
            use super::*;

            macro_rules! value {
                ($name:ident: $data:expr) => {
                    serialize_as!($name:
                        // Serialization started from ElementSerializer::serialize_map
                        BTreeMap::from([("$value", $data)])
                        => "<root/>");
                };
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        // Serialization started from ElementSerializer::serialize_map
                        BTreeMap::from([("$value", $data)])
                        => concat!("<root>", $expected,"</root>"));
                };
            }

            value!(false_: false => "false");
            value!(true_:  true  => "true");

            value!(i8_:    -42i8                => "-42");
            value!(i16_:   -4200i16             => "-4200");
            value!(i32_:   -42000000i32         => "-42000000");
            value!(i64_:   -42000000000000i64   => "-42000000000000");
            value!(isize_: -42000000isize       => "-42000000");

            value!(u8_:    42u8                => "42");
            value!(u16_:   4200u16             => "4200");
            value!(u32_:   42000000u32         => "42000000");
            value!(u64_:   42000000000000u64   => "42000000000000");
            value!(usize_: 42000000usize       => "42000000");

            value!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            value!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            value!(f32_: 4.2f32 => "4.2");
            value!(f64_: 4.2f64 => "4.2");

            value!(char_non_escaped: 'h' => "h");
            value!(char_lt:   '<' => "&lt;");
            value!(char_gt:   '>' => "&gt;");
            value!(char_amp:  '&' => "&amp;");
            value!(char_apos: '\'' => "&apos;");
            value!(char_quot: '"' => "&quot;");
            value!(char_space: ' ' => " ");

            value!(str_non_escaped: "non-escaped string" => "non-escaped string");
            value!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                BTreeMap::from([("$value", Bytes(b"<\"escaped & bytes'>"))])
                => Unsupported("`serialize_bytes` not supported yet"));

            value!(option_none: Option::<&str>::None);
            value!(option_some: Some("non-escaped string") => "non-escaped string");
            value!(option_some_empty_str: Some(""));

            value!(unit: ());
            value!(unit_struct: Unit);
            value!(unit_struct_escaped: UnitEscaped);

            value!(enum_unit: Enum::Unit => "\n  <Unit/>\n");
            err!(enum_unit_escaped:
                BTreeMap::from([("$value", Enum::UnitEscaped)])
                => Unsupported("character `<` is not allowed at the start of an XML name `<\"&'>`"));

            value!(newtype: Newtype(42) => "42");
            value!(enum_newtype: Enum::Newtype(42) => "\n  <Newtype>42</Newtype>\n");

            err!(seq:
                BTreeMap::from([("$value", vec![1, 2, 3])])
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(seq_empty: Vec::<usize>::new());
            err!(tuple:
                BTreeMap::from([("$value", ("<\"&'>", "with\t\n\r spaces", 3usize))])
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            err!(tuple_struct:
                BTreeMap::from([("$value", Tuple("first", 42))])
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(enum_tuple: Enum::Tuple("first", 42)
                => "\n  \
                    <Tuple>first</Tuple>\n  \
                    <Tuple>42</Tuple>\n");

            // We cannot wrap map in any container and should not
            // flatten it, so it is impossible to serialize maps
            err!(map:
                BTreeMap::from([("$value", BTreeMap::from([("_1", 2), ("_3", 4)]))])
                => Unsupported("serialization of map types is not supported in `$value` field"));
            value!(struct_:
                Struct { key: "answer", val: (42, 42) }
                => "\n  \
                    <Struct>\n    \
                        <key>answer</key>\n    \
                        <val>42</val>\n    \
                        <val>42</val>\n  \
                    </Struct>\n");
            value!(enum_struct:
                Enum::Struct { key: "answer", val: (42, 42) }
                => "\n  \
                    <Struct>\n    \
                        <key>answer</key>\n    \
                        <val>42</val>\n    \
                        <val>42</val>\n  \
                    </Struct>\n");
        }

        /// `$value` field inside a struct
        mod struct_ {
            use super::*;

            macro_rules! value {
                ($name:ident: $data:expr => $expected:literal) => {
                    serialize_as!($name:
                        // Serialization started from ElementSerializer::serialize_struct
                        Value {
                            before: "answer",
                            content: $data,
                            after: "answer",
                        }
                        => concat!(
                            "<root>\n  <before>answer</before>",
                            $expected,
                            "<after>answer</after>\n</root>",
                        ));
                };
            }

            value!(false_: false => "false");
            value!(true_:  true  => "true");

            value!(i8_:    -42i8                => "-42");
            value!(i16_:   -4200i16             => "-4200");
            value!(i32_:   -42000000i32         => "-42000000");
            value!(i64_:   -42000000000000i64   => "-42000000000000");
            value!(isize_: -42000000isize       => "-42000000");

            value!(u8_:    42u8                => "42");
            value!(u16_:   4200u16             => "4200");
            value!(u32_:   42000000u32         => "42000000");
            value!(u64_:   42000000000000u64   => "42000000000000");
            value!(usize_: 42000000usize       => "42000000");

            value!(i128_: -420000000000000000000000000000i128 => "-420000000000000000000000000000");
            value!(u128_:  420000000000000000000000000000u128 => "420000000000000000000000000000");

            value!(f32_: 4.2f32 => "4.2");
            value!(f64_: 4.2f64 => "4.2");

            value!(char_non_escaped: 'h' => "h");
            value!(char_lt:   '<' => "&lt;");
            value!(char_gt:   '>' => "&gt;");
            value!(char_amp:  '&' => "&amp;");
            value!(char_apos: '\'' => "&apos;");
            value!(char_quot: '"' => "&quot;");
            value!(char_space: ' ' => " ");

            value!(str_non_escaped: "non-escaped string" => "non-escaped string");
            value!(str_escaped: "<\"escaped & string'>" => "&lt;&quot;escaped &amp; string&apos;&gt;");

            err!(bytes:
                Value {
                    before: "answer",
                    content: Bytes(b"<\"escaped & bytes'>"),
                    after: "answer",
                }
                => Unsupported("`serialize_bytes` not supported yet"));

            value!(option_none: Option::<&str>::None => "");
            value!(option_some: Some("non-escaped string") => "non-escaped string");
            value!(option_some_empty_str: Some("") => "");

            value!(unit: () => "\n  ");
            value!(unit_struct: Unit => "\n  ");
            value!(unit_struct_escaped: UnitEscaped => "\n  ");

            value!(enum_unit: Enum::Unit => "\n  <Unit/>\n  ");
            err!(enum_unit_escaped:
                Value {
                    before: "answer",
                    content: Enum::UnitEscaped,
                    after: "answer",
                }
                => Unsupported("character `<` is not allowed at the start of an XML name `<\"&'>`"));

            value!(newtype: Newtype(42) => "42");
            value!(enum_newtype: Enum::Newtype(42) => "\n  <Newtype>42</Newtype>\n  ");

            err!(seq:
                Value {
                    before: "answer",
                    content: vec![1, 2, 3],
                    after: "answer",
                }
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(seq_empty: Vec::<usize>::new() => "");
            err!(tuple:
                Value {
                    before: "answer",
                    content: ("<\"&'>", "with\t\n\r spaces", 3usize),
                    after: "answer",
                }
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            err!(tuple_struct:
                Value {
                    before: "answer",
                    content: Tuple("first", 42),
                    after: "answer",
                }
                => Unsupported("consequent primitives would be serialized without delimiter and cannot be deserialized back"));
            value!(enum_tuple: Enum::Tuple("first", 42)
                => "\n  \
                    <Tuple>first</Tuple>\n  \
                    <Tuple>42</Tuple>\n  ");

            // We cannot wrap map in any container and should not
            // flatten it, so it is impossible to serialize maps
            err!(map:
                Value {
                    before: "answer",
                    content: BTreeMap::from([("_1", 2), ("_3", 4)]),
                    after: "answer",
                }
                => Unsupported("serialization of map types is not supported in `$value` field"));
            value!(struct_:
                Value {
                    before: "answer",
                    content: Struct { key: "answer", val: (42, 42) },
                    after: "answer",
                }
                => "\n  \
                    <Value>\n    \
                        <before>answer</before>\n    \
                        <Struct>\n      \
                            <key>answer</key>\n      \
                            <val>42</val>\n      \
                            <val>42</val>\n    \
                        </Struct>\n    \
                        <after>answer</after>\n  \
                    </Value>\n  ");
            value!(enum_struct:
                Enum::Struct { key: "answer", val: (42, 42) }
                => "\n  \
                    <Struct>\n    \
                        <key>answer</key>\n    \
                        <val>42</val>\n    \
                        <val>42</val>\n  \
                    </Struct>\n  ");
        }
    }

    mod attributes {
        use super::*;

        serialize_as!(map_attr: BTreeMap::from([("@key1", 1), ("@key2", 2)])
            => r#"<root key1="1" key2="2"/>"#);
        serialize_as!(map_mixed: BTreeMap::from([("@key1", 1), ("key2", 2)])
            => "<root key1=\"1\">\n  \
                    <key2>2</key2>\n\
                </root>");

        serialize_as!(struct_: Attributes { key: "answer", val: (42, 42) }
            => r#"<root key="answer" val="42 42"/>"#);
        serialize_as!(struct_before: AttributesBefore { key: "answer", val: 42 }
            => "<root key=\"answer\">\n  \
                    <val>42</val>\n\
                </root>");
        serialize_as!(struct_after: AttributesAfter { key: "answer", val: 42 }
            => "<root val=\"42\">\n  \
                    <key>answer</key>\n\
                </root>");

        err!(enum_: Enum::Attributes { key: "answer", val: (42, 42) }
            => Unsupported("cannot serialize enum struct variant `Enum::Attributes`"));

        /// Test for https://github.com/tafia/quick-xml/issues/252
        mod optional {
            use super::*;

            serialize_as!(none:
                OptionalAttributes { a: None, b: None }
                => r#"<root a=""/>"#);
            serialize_as!(some_empty_str:
                OptionalAttributes {
                    a: Some(""),
                    b: Some("")
                }
                => r#"<root a="" b=""/>"#);
            serialize_as!(some_non_empty:
                OptionalAttributes {
                    a: Some("a"),
                    b: Some("b")
                }
                => r#"<root a="a" b="b"/>"#);
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/252
    mod optional {
        use super::*;

        serialize_as!(none:
            OptionalElements { a: None, b: None }
            => "<root>\n  \
                    <a/>\n\
                </root>");
        serialize_as!(some_empty_str:
            OptionalElements {
                a: Some(""),
                b: Some("")
            }
            => "<root>\n  \
                    <a/>\n  \
                    <b/>\n\
                </root>");
        serialize_as!(some_non_empty:
            OptionalElements {
                a: Some("a"),
                b: Some("b")
            }
            => "<root>\n  \
                    <a>a</a>\n  \
                    <b>b</b>\n\
                </root>");
    }
}

mod expand_empty_elements {
    use super::*;

    /// Checks that given `$data` successfully serialized as `$expected`
    macro_rules! serialize_as {
        ($name:ident: $data:expr => $expected:expr) => {
            #[test]
            fn $name() {
                let mut buffer = String::new();
                let ser = ElementSerializer {
                    ser: ContentSerializer {
                        writer: &mut buffer,
                        level: QuoteLevel::Full,
                        indent: Indent::None,
                        write_indent: false,
                        text_format: TextFormat::Text,
                        allow_primitive: true,
                        empty_element_handling: EmptyElementHandling::Expanded,
                    },
                    key: XmlName("root"),
                };

                let result = $data.serialize(ser).expect("serialization should succeed");
                assert_eq!(buffer, $expected);
                assert_eq!(result, WriteResult::Element);
            }
        };
    }

    serialize_as!(option_some_empty: Some("") => "<root></root>");
    serialize_as!(option_some_empty_str: Some("") => "<root></root>");

    serialize_as!(unit: () => "<root></root>");
    serialize_as!(unit_struct: Unit => "<root></root>");
    serialize_as!(unit_struct_escaped: UnitEscaped => "<root></root>");

    serialize_as!(enum_unit: Enum::Unit => "<root>Unit</root>");
    serialize_as!(enum_unit_escaped: Enum::UnitEscaped => "<root>&lt;&quot;&amp;&apos;&gt;</root>");
}
