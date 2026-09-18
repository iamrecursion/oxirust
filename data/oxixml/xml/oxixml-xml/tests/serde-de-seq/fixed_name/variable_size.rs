use super::*;

#[derive(Debug, PartialEq, Deserialize)]
struct List {
    item: Vec<()>,
}

/// Simple case: count of elements matches expected size of sequence,
/// each element has the same name. Successful deserialization expected
#[test]
fn simple() {
    let data: List = from_str(
        r#"
                <root>
                    <item/>
                    <item/>
                    <item/>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: vec![(), (), ()],
        }
    );
}

/// Special case: empty sequence. A struct field cannot tell "the element
/// never appeared" apart from "the element appeared zero times" -- both look
/// identical to `MapAccess`, which only ever calls `next_value_seed` for keys
/// it actually saw. `#[serde(default)]` (or `Option<>`) is the documented
/// workaround: it supplies the empty `Vec` when the field is missing.
#[test]
fn empty() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        #[serde(default)]
        item: Vec<()>,
    }

    let data: List = from_str(r#"<root></root>"#).expect("should be Ok");
    assert_eq!(data, List { item: vec![] });

    let data: List = from_str(r#"<root/>"#).expect("should be Ok");
    assert_eq!(data, List { item: vec![] });
}

/// Special case: one-element sequence
#[test]
fn one_element() {
    let data: List = from_str(
        r#"
                <root>
                    <item/>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(data, List { item: vec![()] });
}

/// Mixed content assumes, that some elements will have an internal
/// name `$text` or `$value`, so, unless field named the same, it is expected
/// to fail
#[test]
fn mixed_content() {
    let data = from_str::<List>(
        r#"
                <root>
                    <element/>
                    text
                    <![CDATA[cdata]]>
                </root>
                "#,
    );

    match data {
        Err(DeError::Custom(e)) => assert_eq!(e, "missing field `item`"),
        e => panic!(
            r#"Expected `Err(Custom("missing field `item`"))`, but got `{:?}`"#,
            e
        ),
    }
}

/// In those tests sequence should be deserialized from the XML
/// with additional elements that is not defined in the struct.
/// That fields should be skipped during deserialization
mod unknown_items {
    use super::*;

    #[test]
    fn before() {
        let data: List = from_str(
            r#"
                    <root>
                        <unknown/>
                        <item/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            List {
                item: vec![(), (), ()],
            }
        );
    }

    #[test]
    fn after() {
        let data: List = from_str(
            r#"
                    <root>
                        <item/>
                        <item/>
                        <item/>
                        <unknown/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            List {
                item: vec![(), (), ()],
            }
        );
    }

    #[test]
    fn overlapped() {
        let data = from_str::<List>(
            r#"
                    <root>
                        <item/>
                        <unknown/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            List {
                item: vec![(), (), ()],
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `item`"),
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `item`"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/435
    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Root {
            outer: Vec<List>,
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <outer><item/></outer>
                        <unknown/>
                        <outer><item/></outer>
                        <outer><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                outer: vec![
                    List { item: vec![()] },
                    List { item: vec![()] },
                    List { item: vec![()] },
                ],
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `outer`"),
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `outer`"))`, but got `{:?}`"#,
                e
            ),
        }
    }
}

/// In those tests non-sequential field is defined in the struct
/// before sequential, so it will be deserialized before the list.
/// That struct should be deserialized from the XML where these
/// fields comes in an arbitrary order
mod field_before_list {
    use super::*;

    #[derive(Debug, PartialEq, Default, Deserialize)]
    struct Root {
        node: (),
        item: Vec<()>,
    }

    #[test]
    fn before() {
        let data: Root = from_str(
            r#"
                    <root>
                        <node/>
                        <item/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                node: (),
                item: vec![(), (), ()],
            }
        );
    }

    #[test]
    fn after() {
        let data: Root = from_str(
            r#"
                    <root>
                        <item/>
                        <item/>
                        <item/>
                        <node/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                node: (),
                item: vec![(), (), ()],
            }
        );
    }

    #[test]
    fn overlapped() {
        let data = from_str::<Root>(
            r#"
                    <root>
                        <item/>
                        <node/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                node: (),
                item: vec![(), (), ()],
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "duplicate field `item`")
            }
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `item`"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/435
    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Root {
            node: (),
            outer: Vec<List>,
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <outer><item/></outer>
                        <node/>
                        <outer><item/></outer>
                        <outer><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                node: (),
                outer: vec![
                    List { item: vec![()] },
                    List { item: vec![()] },
                    List { item: vec![()] },
                ],
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `outer`"),
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `outer`"))`, but got `{:?}`"#,
                e
            ),
        }
    }
}

/// In those tests non-sequential field is defined in the struct
/// after sequential, so it will be deserialized after the list.
/// That struct should be deserialized from the XML where these
/// fields comes in an arbitrary order
mod field_after_list {
    use super::*;

    #[derive(Debug, PartialEq, Default, Deserialize)]
    struct Root {
        item: Vec<()>,
        node: (),
    }

    #[test]
    fn before() {
        let data: Root = from_str(
            r#"
                    <root>
                        <node/>
                        <item/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                item: vec![(), (), ()],
                node: (),
            }
        );
    }

    #[test]
    fn after() {
        let data: Root = from_str(
            r#"
                    <root>
                        <item/>
                        <item/>
                        <item/>
                        <node/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                item: vec![(), (), ()],
                node: (),
            }
        );
    }

    #[test]
    fn overlapped() {
        let data = from_str::<Root>(
            r#"
                    <root>
                        <item/>
                        <node/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                item: vec![(), (), ()],
                node: (),
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "duplicate field `item`")
            }
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `item`"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/435
    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Root {
            outer: Vec<List>,
            node: (),
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <outer><item/></outer>
                        <node/>
                        <outer><item/></outer>
                        <outer><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                outer: vec![
                    List { item: vec![()] },
                    List { item: vec![()] },
                    List { item: vec![()] },
                ],
                node: (),
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `outer`"),
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `outer`"))`, but got `{:?}`"#,
                e
            ),
        }
    }
}

/// In those tests two lists are deserialized simultaneously.
/// Lists should be deserialized even when them overlaps
mod two_lists {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Pair {
        item: Vec<()>,
        element: Vec<()>,
    }

    #[test]
    fn splitted() {
        let data: Pair = from_str(
            r#"
                    <root>
                        <element/>
                        <element/>
                        <item/>
                        <item/>
                        <item/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Pair {
                item: vec![(), (), ()],
                element: vec![(), ()],
            }
        );
    }

    #[test]
    fn overlapped() {
        let data = from_str::<Pair>(
            r#"
                    <root>
                        <item/>
                        <element/>
                        <item/>
                        <element/>
                        <item/>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Pair {
                item: vec![(), (), ()],
                element: vec![(), ()],
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `item`"),
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `item`"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Pair {
            outer: Vec<List>,
            element: Vec<()>,
        }

        let data = from_str::<Pair>(
            r#"
                    <root>
                        <outer><item/></outer>
                        <element/>
                        <outer><item/></outer>
                        <outer><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Pair {
                outer: vec![
                    List { item: vec![()] },
                    List { item: vec![()] },
                    List { item: vec![()] },
                ],
                element: vec![()],
            }
        );

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `outer`"),
            e => panic!(
                r#"Expected `Err(Custom("duplicate field `outer`"))`, but got `{:?}`"#,
                e
            ),
        }
    }
}

/// Deserialization of primitives slightly differs from deserialization
/// of complex types, so need to check this separately
#[test]
fn primitives() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        item: Vec<usize>,
    }

    let data: List = from_str(
        r#"
                <root>
                    <item>41</item>
                    <item>42</item>
                    <item>43</item>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: vec![41, 42, 43],
        }
    );

    from_str::<List>(
        r#"
                <root>
                    <item>41</item>
                    <item><item>42</item></item>
                    <item>43</item>
                </root>
                "#,
    )
    .expect_err("should be Err");
}

/// This test ensures that composition of deserializer building blocks
/// plays well
#[test]
fn list_of_struct() {
    #[derive(Debug, PartialEq, Default, Deserialize)]
    #[serde(default)]
    struct Struct {
        #[serde(rename = "@attribute")]
        attribute: Option<String>,
        element: Option<String>,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        item: Vec<Struct>,
    }

    let data: List = from_str(
        r#"
                <root>
                    <item/>
                    <item attribute="value"/>
                    <item>
                        <element>value</element>
                    </item>
                    <item attribute="value">
                        <element>value</element>
                    </item>
                </root>
                "#,
    )
    .expect("should be Ok");
    assert_eq!(
        data,
        List {
            item: vec![
                Struct {
                    attribute: None,
                    element: None,
                },
                Struct {
                    attribute: Some("value".to_string()),
                    element: None,
                },
                Struct {
                    attribute: None,
                    element: Some("value".to_string()),
                },
                Struct {
                    attribute: Some("value".to_string()),
                    element: Some("value".to_string()),
                },
            ],
        }
    );
}

#[test]
fn text_and_value() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        #[serde(rename = "$text")]
        text: (),
        item: Vec<()>,
    }

    let data: List = from_str(
        r#"
                <root>
                    <item/>
                    <item/>
                    text
                    <![CDATA[cdata]]>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            text: (),
            item: vec![(); 2],
        }
    );
}

/// Checks that sequences represented by elements can contain sequences,
/// represented by `xs:list`s
mod xs_list {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct List {
        /// Outer list mapped to elements, inner -- to `xs:list`.
        ///
        /// `#[serde(default)]` is required to correctly deserialize
        /// empty sequence, because without elements the field
        /// also is missing and derived `Deserialize` implementation
        /// would complain about that unless field is marked as
        /// `default`.
        #[serde(default)]
        item: Vec<Vec<String>>,
    }

    /// Special case: zero elements
    #[test]
    fn zero() {
        let data: List = from_str(
            r#"
                    <root>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(data, List { item: vec![] });
    }

    /// Special case: one element
    #[test]
    fn one() {
        let data: List = from_str(
            r#"
                    <root>
                        <item>first list</item>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            List {
                item: vec![vec!["first".to_string(), "list".to_string()]]
            }
        );
    }

    /// Special case: empty `xs:list`
    #[test]
    fn empty() {
        let data: List = from_str(
            r#"
                    <root>
                        <item/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(data, List { item: vec![vec![]] });
    }

    /// Special case: outer list is always mapped to an elements sequence,
    /// not to an `xs:list`
    #[test]
    fn element() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct List {
            /// List mapped to elements, inner -- to `xs:list`.
            ///
            /// `#[serde(default)]` is not required, because correct
            /// XML will always contains at least 1 element.
            item: Vec<String>,
        }

        let data: List = from_str(
            r#"
                    <root>
                        <item>first item</item>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            List {
                item: vec!["first item".to_string()]
            }
        );
    }

    /// This tests demonstrates, that for `$value` field (`list`) actual
    /// name of XML element (`item`) does not matter. That allows list
    /// item to be an enum, where tag name determines enum variant
    #[test]
    fn many() {
        let data: List = from_str(
            r#"
                    <root>
                        <item>first list</item>
                        <item>second list</item>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            List {
                item: vec![
                    vec!["first".to_string(), "list".to_string()],
                    vec!["second".to_string(), "list".to_string()],
                ]
            }
        );
    }
}
