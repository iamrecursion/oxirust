use super::*;

#[derive(Debug, PartialEq, Deserialize)]
struct List<T = ()> {
    item: [T; 3],
}

/// Simple case: count of elements matches expected size of sequence,
/// each element has the same name. Successful deserialization expected
#[test]
fn simple() {
    from_str::<List>(
        r#"
                <root>
                    <item/>
                    <item/>
                    <item/>
                </root>
                "#,
    )
    .expect("should be Ok");
}

/// Special case: empty sequence. A struct field cannot tell "the element
/// never appeared" apart from "the element appeared zero times" -- both look
/// identical to `MapAccess`, which only ever calls `next_value_seed` for keys
/// it actually saw. `#[serde(default)]` (or `Option<>`) is the documented
/// workaround: it supplies the empty array when the field is missing, exactly
/// as `[(); 0]` already does unconditionally through `Default`.
#[test]
fn empty() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        #[serde(default)]
        item: [(); 0],
    }

    assert_eq!(
        from_str::<List>(r#"<root></root>"#).expect("should be Ok"),
        List { item: [] }
    );
    assert_eq!(
        from_str::<List>(r#"<root/>"#).expect("should be Ok"),
        List { item: [] }
    );
}

/// Special case: one-element sequence
#[test]
fn one_element() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        item: [(); 1],
    }

    from_str::<List>(
        r#"
                <root>
                    <item/>
                </root>
                "#,
    )
    .expect("should be Ok");
}

/// Fever elements than expected size of sequence, each element has
/// the same name. Failure expected
#[test]
fn fever_elements() {
    let data = from_str::<List>(
        r#"
                <root>
                    <item/>
                    <item/>
                </root>
                "#,
    );

    match data {
        Err(DeError::Custom(e)) => {
            assert_eq!(e, "invalid length 2, expected an array of length 3")
        }
        e => panic!(
            r#"Expected `Err(Custom("invalid length 2, expected an array of length 3"))`, but got `{:?}`"#,
            e
        ),
    }
}

/// More elements than expected size of sequence, each element has
/// the same name. Failure expected. If you wish to ignore excess
/// elements, use the special type, that consume as much elements
/// as possible, but ignores excess elements
#[test]
fn excess_elements() {
    let data = from_str::<List>(
        r#"
                <root>
                    <item/>
                    <item/>
                    <item/>
                    <item/>
                </root>
                "#,
    );

    match data {
        Err(DeError::Custom(e)) => assert_eq!(e, "duplicate field `item`"),
        e => panic!(
            r#"Expected `Err(Custom("duplicate field `item`"))`, but got `{:?}`"#,
            e
        ),
    }
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

/// In those tests sequence should be deserialized from an XML
/// with additional elements that is not defined in the struct.
/// That fields should be skipped during deserialization
mod unknown_items {
    use super::*;

    #[test]
    fn before() {
        from_str::<List>(
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
    }

    #[test]
    fn after() {
        from_str::<List>(
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
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/435
    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Root {
            outer: [List; 3],
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <outer><item/><item/><item/></outer>
                        <unknown/>
                        <outer><item/><item/><item/></outer>
                        <outer><item/><item/><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
                e
            ),
        }
    }
}

/// In those tests non-sequential field is defined in the struct
/// before sequential, so it will be deserialized before the list.
/// That struct should be deserialized from an XML where these
/// fields comes in an arbitrary order
mod field_before_list {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Root {
        node: (),
        item: [(); 3],
    }

    #[test]
    fn before() {
        from_str::<Root>(
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
    }

    #[test]
    fn after() {
        from_str::<Root>(
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
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
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
            outer: [List; 3],
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <outer><item/><item/><item/></outer>
                        <node/>
                        <outer><item/><item/><item/></outer>
                        <outer><item/><item/><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
                e
            ),
        }
    }
}

/// In those tests non-sequential field is defined in the struct
/// after sequential, so it will be deserialized after the list.
/// That struct should be deserialized from an XML where these
/// fields comes in an arbitrary order
mod field_after_list {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Root {
        item: [(); 3],
        node: (),
    }

    #[test]
    fn before() {
        from_str::<Root>(
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
    }

    #[test]
    fn after() {
        from_str::<Root>(
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
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/435
    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Root {
            outer: [List; 3],
            node: (),
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <outer><item/><item/><item/></outer>
                        <node/>
                        <outer><item/><item/><item/></outer>
                        <outer><item/><item/><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
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
        item: [(); 3],
        element: [(); 2],
    }

    #[test]
    fn splitted() {
        from_str::<Pair>(
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
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
                e
            ),
        }
    }

    /// Test for https://github.com/tafia/quick-xml/issues/435
    #[test]
    fn overlapped_with_nested_list() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Pair {
            outer: [List; 3],
            element: [(); 2],
        }

        let data = from_str::<Pair>(
            r#"
                    <root>
                        <outer><item/><item/><item/></outer>
                        <element/>
                        <outer><item/><item/><item/></outer>
                        <element/>
                        <outer><item/><item/><item/></outer>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        data.expect("should be Ok");

        #[cfg(not(feature = "overlapped-lists"))]
        match data {
            Err(DeError::Custom(e)) => {
                assert_eq!(e, "invalid length 1, expected an array of length 3")
            }
            e => panic!(
                r#"Expected `Err(Custom("invalid length 1, expected an array of length 3"))`, but got `{:?}`"#,
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
        item: [usize; 3],
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
    assert_eq!(data, List { item: [41, 42, 43] });

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
        item: [Struct; 4],
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
            item: [
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
fn list_of_list() {
    let data: List<Vec<String>> = from_str(
        r#"
                <root>
                    <item>first item</item>
                    <item>second item</item>
                    <item>third item</item>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: [
                vec!["first".to_string(), "item".to_string()],
                vec!["second".to_string(), "item".to_string()],
                vec!["third".to_string(), "item".to_string()],
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
        item: [(); 2],
    }

    from_str::<List>(
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
}

/// Checks that sequences represented by elements can contain sequences,
/// represented by [`xs:list`s](https://www.w3schools.com/xml/el_list.asp)
mod xs_list {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct List {
        /// Outer list mapped to elements, inner -- to `xs:list`.
        ///
        /// `#[serde(default)]` is not required, because correct
        /// XML will always contains at least 1 element.
        item: [Vec<String>; 1],
    }

    /// Special case: zero elements
    #[test]
    fn zero() {
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
            item: [Vec<String>; 0],
        }

        let data: List = from_str(
            r#"
                    <root>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(data, List { item: [] });
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
                item: [vec!["first".to_string(), "list".to_string()]]
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

        assert_eq!(data, List { item: [vec![]] });
    }

    /// Special case: outer list is always mapped to an elements sequence,
    /// not to an `xs:list`
    #[test]
    fn element() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct List {
            /// Outer list mapped to elements, inner -- to `xs:list`.
            ///
            /// `#[serde(default)]` is not required, because correct
            /// XML will always contains at least 1 element.
            item: [String; 1],
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
                item: ["first item".to_string()]
            }
        );
    }

    /// This tests demonstrates, that for `$value` field (`list`) actual
    /// name of XML element (`item`) does not matter. That allows list
    /// item to be an enum, where tag name determines enum variant
    #[test]
    fn many() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct List {
            /// Outer list mapped to elements, inner -- to `xs:list`.
            ///
            /// `#[serde(default)]` is not required, because correct
            /// XML will always contains at least 1 element.
            item: [Vec<String>; 2],
        }

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
                item: [
                    vec!["first".to_string(), "list".to_string()],
                    vec!["second".to_string(), "list".to_string()],
                ]
            }
        );
    }
}
