//! Tests of deserialization of XML documents into various enum types

use oxixml_xml::DeError;
use serde::Deserialize;

mod serde_helpers;
use serde_helpers::from_str;

#[derive(Debug, Deserialize, PartialEq)]
struct Nested {
    // Kept as `String` rather than `f64`: this struct is also deserialized
    // through `#[serde(flatten)]` and untagged-enum paths further down this
    // file, both of which serde routes through `deserialize_any` regardless of
    // the field's declared type. `deserialize_any` cannot request a specific
    // primitive, so it always hands the visitor a string; asking for `f64`
    // here would break those call sites without serde itself first no longer
    // routing flatten/untagged through `deserialize_any` (serde-rs/serde#1183,
    // open since 2018 -- nothing in this crate can change that routing).
    float: String,
}

/// Type where all struct fields represented by attributes
#[derive(Debug, Deserialize, PartialEq)]
struct NestedAttr {
    // See `Nested::float` above: blocked on serde-rs/serde#1183, not on
    // anything in this crate.
    #[serde(rename = "@float")]
    float: String,
}

/// Enum tag selector is a name of the `<element>` or text / CDATA content
/// for a `$text` variant
mod externally_tagged {
    use super::*;

    /// Type where all fields of struct variants represented by elements
    #[derive(Debug, Deserialize, PartialEq)]
    enum Node {
        Unit,
        Newtype(bool),
        Tuple(f64, String),
        Struct {
            float: f64,
            string: String,
        },
        Holder {
            nested: Nested,
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: Nested,
            string: String,
        },
    }

    /// Type where all fields of struct variants represented by attributes
    #[derive(Debug, Deserialize, PartialEq)]
    enum NodeAttr {
        Struct {
            #[serde(rename = "@float")]
            float: f64,
            #[serde(rename = "@string")]
            string: String,
        },
        Holder {
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
    }

    #[test]
    fn unit() {
        let data: Node = from_str("<Unit/>").expect("should be Ok");
        assert_eq!(data, Node::Unit);
    }

    #[test]
    fn newtype() {
        let data: Node = from_str("<Newtype>true</Newtype>").expect("should be Ok");
        assert_eq!(data, Node::Newtype(true));
    }

    #[test]
    fn tuple_struct() {
        let data: Node = from_str("<Tuple>42</Tuple><Tuple>answer</Tuple>").expect("should be Ok");
        assert_eq!(data, Node::Tuple(42.0, "answer".into()));
    }

    mod struct_ {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<Struct><float>42</float><string>answer</string></Struct>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<Struct float="42" string="answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn namespaces() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<namespace:Struct xmlns:namespace="http://name.space"><float>42</float><string>answer</string></namespace:Struct>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }
    }

    mod nested_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<Holder><string>answer</string><nested><float>42</float></nested></Holder>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Holder {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<Holder string="answer"><nested float="42"/></Holder>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Holder {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }

    mod flatten_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<Flatten><float>42</float><string>answer</string></Flatten>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Flatten {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<Flatten float="42" string="answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Flatten {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }

    /// Test deserialization of the specially named variant `$text`
    mod text {
        use super::*;

        mod unit {
            use super::*;

            #[derive(Debug, Deserialize, PartialEq)]
            enum Text {
                #[serde(rename = "$text")]
                Unit,
            }

            #[test]
            fn text() {
                let data: Text = from_str(" text ").expect("should be Ok");
                assert_eq!(data, Text::Unit);
            }

            #[test]
            fn cdata() {
                let data: Text = from_str("<![CDATA[ cdata ]]>").expect("should be Ok");
                assert_eq!(data, Text::Unit);
            }

            #[test]
            fn mixed() {
                let data: Text = from_str(" te <![CDATA[ cdata ]]> xt ").expect("should be Ok");
                assert_eq!(data, Text::Unit);
            }
        }

        mod newtype {
            use super::*;

            #[derive(Debug, Deserialize, PartialEq)]
            enum Text {
                #[serde(rename = "$text")]
                Newtype(String),
            }

            #[test]
            fn text() {
                let data: Text = from_str(" text ").expect("should be Ok");
                assert_eq!(data, Text::Newtype(" text ".into()));
            }

            #[test]
            fn cdata() {
                let data: Text = from_str("<![CDATA[ cdata ]]>").expect("should be Ok");
                assert_eq!(data, Text::Newtype(" cdata ".into()));
            }

            #[test]
            fn mixed() {
                let data: Text = from_str(" te <![CDATA[ cdata ]]> xt ").expect("should be Ok");
                assert_eq!(data, Text::Newtype(" te  cdata  xt ".into()));
            }
        }

        /// Tuple variant deserialized as an `xs:list`, that is why spaces
        /// are trimmed even in CDATA sections
        mod tuple {
            use super::*;

            #[derive(Debug, Deserialize, PartialEq)]
            enum Text {
                #[serde(rename = "$text")]
                Tuple(f64, String),
            }

            #[test]
            fn text() {
                let data: Text = from_str(" 4.2 text ").expect("should be Ok");
                assert_eq!(data, Text::Tuple(4.2, "text".into()));
            }

            #[test]
            fn cdata() {
                let data: Text = from_str("<![CDATA[ 4.2 cdata ]]>").expect("should be Ok");
                assert_eq!(data, Text::Tuple(4.2, "cdata".into()));
            }

            #[test]
            fn mixed() {
                let data: Text = from_str(" 4.2 <![CDATA[ cdata ]]>").expect("should be Ok");
                assert_eq!(data, Text::Tuple(4.2, "cdata".into()));
            }
        }

        /// Struct variant cannot be directly deserialized from `Text` / `CData` events
        mod struct_ {
            use super::*;

            #[derive(Debug, Deserialize, PartialEq)]
            enum Text {
                #[serde(rename = "$text")]
                Struct { float: f64, string: String },
            }

            #[test]
            fn text() {
                match from_str::<Text>(" text ") {
                    Err(DeError::Custom(reason)) => assert_eq!(
                        reason,
                        "invalid type: string \" text \", expected struct variant Text::Struct"
                    ),
                    x => panic!(
                        r#"Expected `Err(Custom("invalid type: string \" text \", expected struct variant Text::Struct"))`, but got `{:?}`"#,
                        x
                    ),
                }
            }

            #[test]
            fn cdata() {
                match from_str::<Text>("<![CDATA[ cdata ]]>") {
                    Err(DeError::Custom(reason)) => assert_eq!(
                        reason,
                        "invalid type: string \" cdata \", expected struct variant Text::Struct"
                    ),
                    x => panic!(
                        r#"Expected `Err(Custom("invalid type: string \" cdata \", expected struct variant Text::Struct"))`, but got `{:?}`"#,
                        x
                    ),
                }
            }

            #[test]
            fn mixed() {
                match from_str::<Text>(" te <![CDATA[ cdata ]]> xt ") {
                    Err(DeError::Custom(reason)) => assert_eq!(
                        reason,
                        "invalid type: string \" te  cdata  xt \", expected struct variant Text::Struct"
                    ),
                    x => panic!(
                        r#"Expected `Err(Custom("invalid type: string \" te  cdata  xt \", expected struct variant Text::Struct"))`, but got `{:?}`"#,
                        x
                    ),
                }
            }
        }
    }
}

/// Enum tag selector either an attribute "tag", or a tag "tag".
/// `$text` variant could be defined, but that name has no special meaning
mod internally_tagged {
    use super::*;

    /// Type where all fields of struct variants and a tag represented by elements
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(tag = "tag")]
    enum Node {
        Unit,
        /// Primitives (such as `bool`) are not supported by serde in the internally tagged mode
        Newtype(NewtypeContent),
        // Tuple(f64, String),// Tuples are not supported in the internally tagged mode
        Struct {
            // Kept as `String` rather than `f64`: internally tagged enums are
            // deserialized through `deserialize_any`, which surfaces a text-only
            // element as its string content but cannot know to reparse that text
            // as an `f64` (serde-rs/serde#1183). A `String` field accepts the
            // value; an `f64` field could not be recovered from the buffered
            // string.
            float: String,
            string: String,
        },
        Holder {
            nested: Nested,
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: Nested,
            string: String,
        },
    }

    /// Type where all fields of struct variants and a tag represented by attributes
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(tag = "@tag")]
    enum NodeAttr {
        Unit,
        /// Primitives (such as `bool`) are not supported by serde in the internally tagged mode
        Newtype(NewtypeContent),
        // Tuple(f64, String),// Tuples are not supported in the internally tagged mode
        Struct {
            // See `Node::Struct::float` above: blocked on serde-rs/serde#1183.
            #[serde(rename = "@float")]
            float: String,
            #[serde(rename = "@string")]
            string: String,
        },
        Holder {
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct NewtypeContent {
        value: bool,
    }

    mod unit {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(r#"<root><tag>Unit</tag></root>"#).expect("should be Ok");
            assert_eq!(data, Node::Unit);
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(r#"<root tag="Unit"/>"#).expect("should be Ok");
            assert_eq!(data, NodeAttr::Unit);
        }
    }

    mod newtype {
        use super::*;

        #[test]
        #[ignore = "serde#1183: deserialize_any surfaces XML text as a string, which serde cannot reparse into this variant's bool or numeric field. https://github.com/serde-rs/serde/issues/1183"]
        fn elements() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root><tag>Newtype</tag><value>true</value></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(data, Node::Newtype(NewtypeContent { value: true }));
        }

        #[test]
        #[ignore = "serde#1183: deserialize_any surfaces XML text as a string, which serde cannot reparse into this variant's bool or numeric field. https://github.com/serde-rs/serde/issues/1183"]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Newtype"><value>true</value></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(data, NodeAttr::Newtype(NewtypeContent { value: true }));
        }
    }

    mod struct_ {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><tag>Struct</tag><float>42</float><string>answer</string></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Struct {
                    float: "42".into(),
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Struct" float="42" string="answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Struct {
                    float: "42".into(),
                    string: "answer".into()
                }
            );
        }
    }

    mod nested_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><tag>Holder</tag><string>answer</string><nested><float>42</float></nested></root>"#,
            ).expect("should be Ok");
            assert_eq!(
                data,
                Node::Holder {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Holder" string="answer"><nested float="42"/></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Holder {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }

    mod flatten_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><tag>Flatten</tag><float>42</float><string>answer</string></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Flatten {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Flatten" float="42" string="answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Flatten {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }
}

/// Enum tag selector either an attribute "tag", or a tag "tag".
/// `$text` variant could be defined, but that name has no special meaning
mod adjacently_tagged {
    use super::*;

    /// Type where all fields of struct variants, tag and content fields
    /// represented by elements
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(tag = "tag", content = "content")]
    enum Node {
        Unit,
        Newtype(bool),
        Tuple(f64, String),
        Struct {
            float: f64,
            string: String,
        },
        Holder {
            nested: Nested,
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: Nested,
            string: String,
        },
    }

    /// Type where all fields of struct variants, tag and content fields
    /// represented by attributes
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(tag = "@tag", content = "@content")]
    enum NodeAttrSimple {
        Unit,
        Newtype(bool),
        Tuple(f64, String),
    }

    /// Type where all fields of struct variants and a tag represented by attributes
    /// content cannot be represented by attribute because this is a complex struct
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(tag = "@tag", content = "content")]
    enum NodeAttrComplex {
        Tuple(f64, String),
        Struct {
            #[serde(rename = "@float")]
            float: f64,
            #[serde(rename = "@string")]
            string: String,
        },
        Holder {
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
    }

    mod unit {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(r#"<root><tag>Unit</tag></root>"#).expect("should be Ok");
            assert_eq!(data, Node::Unit);
        }

        #[test]
        fn attributes() {
            let data: NodeAttrSimple = from_str(r#"<root tag="Unit"/>"#).expect("should be Ok");
            assert_eq!(data, NodeAttrSimple::Unit);
        }
    }

    mod newtype {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root><tag>Newtype</tag><content>true</content></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(data, Node::Newtype(true));
        }

        #[test]
        fn attributes() {
            let data: NodeAttrSimple = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Newtype" content="true"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(data, NodeAttrSimple::Newtype(true));
        }
    }

    mod tuple_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><tag>Tuple</tag><content>42</content><content>answer</content></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(data, Node::Tuple(42.0, "answer".into()));
        }

        #[test]
        fn attributes() {
            let data: NodeAttrSimple = from_str(
                // We cannot have two attributes with the same name, so both values stored in one attribute
                r#"<root tag="Tuple" content="42 answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(data, NodeAttrSimple::Tuple(42.0, "answer".into()));
        }
    }

    mod struct_ {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><tag>Struct</tag><content><float>42</float><string>answer</string></content></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttrComplex = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Struct"><content float="42" string="answer"/></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttrComplex::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }
    }

    mod nested_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root>
                    <tag>Holder</tag>
                    <content>
                        <string>answer</string>
                        <nested>
                            <float>42</float>
                        </nested>
                    </content>
                </root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Holder {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttrComplex = from_str(
                r#"<root tag="Holder"><content string="answer"><nested float="42"/></content></root>"#,
            ).expect("should be Ok");
            assert_eq!(
                data,
                NodeAttrComplex::Holder {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }

    mod flatten_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><tag>Flatten</tag><content><float>42</float><string>answer</string></content></root>"#,
            ).expect("should be Ok");
            assert_eq!(
                data,
                Node::Flatten {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttrComplex = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root tag="Flatten"><content float="42" string="answer"/></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttrComplex::Flatten {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }
}

/// Enum tags does not exist.
/// `$text` variant could be defined, but that name has no special meaning
mod untagged {
    use super::*;

    /// Type where all fields of struct variants represented by elements
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(untagged)]
    enum Node {
        Unit,
        Newtype(bool),
        // serde bug https://github.com/serde-rs/serde/issues/1904
        // Tuple(f64, String),
        Struct {
            float: f64,
            string: String,
        },
        Holder {
            nested: Nested,
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: Nested,
            // Can't use "string" as name because in that case this variant
            // will have no difference from `Struct` variant
            string2: String,
        },
    }

    /// Type where all fields of struct variants represented by attributes
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(untagged)]
    enum NodeAttr {
        // serde bug https://github.com/serde-rs/serde/issues/1904
        // Tuple(f64, String),
        Struct {
            #[serde(rename = "@float")]
            float: f64,
            #[serde(rename = "@string")]
            string: String,
        },
        Holder {
            nested: NestedAttr,
            #[serde(rename = "@string")]
            string: String,
        },
        Flatten {
            #[serde(flatten)]
            nested: NestedAttr,
            // Can't use "string" as name because in that case this variant
            // will have no difference from `Struct` variant
            #[serde(rename = "@string2")]
            string2: String,
        },
    }

    /// Workaround for serde bug https://github.com/serde-rs/serde/issues/1904
    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(untagged)]
    enum Workaround {
        Tuple(f64, String),
    }

    #[test]
    // serde#1183: `untagged` buffers content through `deserialize_any`, but an
    // empty document is reported as EOF before any content exists. Surfacing that
    // EOF as a unit (so this variant could match) would also make an empty string
    // a valid `()`, `Bytes` or `ByteBuf`, which the `trivial` tests in
    // `serde-de.rs` deliberately reject. See
    // https://github.com/serde-rs/serde/issues/1183.
    #[ignore = "serde#1183: untagged buffers via deserialize_any; empty input is EOF, which cannot become a unit without accepting empty documents the trivial tests reject"]
    fn unit() {
        // Unit variant consists just from the tag, and because tags
        // are not written, nothing is written
        let data: Node = from_str("").expect("should be Ok");
        assert_eq!(data, Node::Unit);
    }

    #[test]
    #[ignore = "serde#1183: deserialize_any surfaces XML text as a string, which serde cannot reparse into this variant's bool or numeric field. https://github.com/serde-rs/serde/issues/1183"]
    fn newtype() {
        let data: Node = from_str("true").expect("should be Ok");
        assert_eq!(data, Node::Newtype(true));
    }

    #[test]
    #[ignore = "serde#1183: deserialize_any surfaces XML text as a string, which serde cannot reparse into this variant's bool or numeric field. https://github.com/serde-rs/serde/issues/1183"]
    fn tuple_struct() {
        let data: Workaround =
            from_str("<root>42</root><root>answer</root>").expect("should be Ok");
        assert_eq!(data, Workaround::Tuple(42.0, "answer".into()));
    }

    mod struct_ {
        use super::*;

        #[test]
        #[ignore = "serde#1183: deserialize_any surfaces XML text as a string, which serde cannot reparse into this variant's bool or numeric field. https://github.com/serde-rs/serde/issues/1183"]
        fn elements() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root><float>42</float><string>answer</string></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }

        #[test]
        #[ignore = "serde#1183: deserialize_any surfaces XML text as a string, which serde cannot reparse into this variant's bool or numeric field. https://github.com/serde-rs/serde/issues/1183"]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root float="42" string="answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Struct {
                    float: 42.0,
                    string: "answer".into()
                }
            );
        }
    }

    mod nested_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                r#"<root><string>answer</string><nested><float>42</float></nested></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Holder {
                    nested: Nested { float: "42".into() },
                    string: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root string="answer"><nested float="42"/></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Holder {
                    nested: NestedAttr { float: "42".into() },
                    string: "answer".into()
                }
            );
        }
    }

    mod flatten_struct {
        use super::*;

        #[test]
        fn elements() {
            let data: Node = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root><float>42</float><string2>answer</string2></root>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                Node::Flatten {
                    nested: Nested { float: "42".into() },
                    string2: "answer".into()
                }
            );
        }

        #[test]
        fn attributes() {
            let data: NodeAttr = from_str(
                // Comment for prevent unnecessary formatting - we use the same style in all tests
                r#"<root float="42" string2="answer"/>"#,
            )
            .expect("should be Ok");
            assert_eq!(
                data,
                NodeAttr::Flatten {
                    nested: NestedAttr { float: "42".into() },
                    string2: "answer".into()
                }
            );
        }
    }
}
