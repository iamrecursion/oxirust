use super::*;

#[derive(Debug, PartialEq, Deserialize)]
struct List<T = Choice> {
    #[serde(rename = "$value")]
    item: [T; 3],
}

/// Simple case: count of elements matches expected size of sequence,
/// each element has the same name. Successful deserialization expected
#[test]
fn simple() {
    let data: List = from_str(
        r#"
                <root>
                    <one/>
                    <two/>
                    <three/>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: [Choice::One, Choice::Two, Choice::Other("three".into())],
        }
    );
}

/// Special case: empty sequence. A struct field cannot tell "the element
/// never appeared" apart from "the element appeared zero times" -- both look
/// identical to `MapAccess`, which only ever calls `next_value_seed` for keys
/// it actually saw. `#[serde(default)]` (or `Option<>`) is the documented
/// workaround: it supplies the empty array when the field is missing, exactly
/// as `[Choice; 0]` already does unconditionally through `Default`.
#[test]
fn empty() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        #[serde(default)]
        #[serde(rename = "$value")]
        item: [Choice; 0],
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
        #[serde(rename = "$value")]
        item: [Choice; 1],
    }

    let data: List = from_str(
        r#"
                <root>
                    <one/>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: [Choice::One],
        }
    );
}

/// Fever elements than expected size of sequence, each element has
/// the same name. Failure expected
#[test]
fn fever_elements() {
    from_str::<List>(
        r#"
                <root>
                    <one/>
                    <two/>
                </root>
                "#,
    )
    .expect_err("should be Err");
}

/// More elements than expected size of sequence, each element has
/// the same name. Failure expected. If you wish to ignore excess
/// elements, use the special type, that consume as much elements
/// as possible, but ignores excess elements
#[test]
fn excess_elements() {
    from_str::<List>(
        r#"
                <root>
                    <one/>
                    <two/>
                    <three/>
                    <four/>
                </root>
                "#,
    )
    .expect_err("should be Err");
}

#[test]
fn mixed_content() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        /// Text and CDATA represents a one logical text item
        #[serde(rename = "$value")]
        item: [(); 2],
    }

    from_str::<List>(
        r#"
                <root>
                    <element/>
                    text
                    <![CDATA[cdata]]>
                </root>
                "#,
    )
    .expect("should be Ok");
}

// There cannot be unknown items, because any tag name is accepted

/// In those tests non-sequential field is defined in the struct
/// before sequential, so it will be deserialized before the list.
/// That struct should be deserialized from the XML where these
/// fields comes in an arbitrary order
mod field_before_list {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Root {
        node: (),
        #[serde(rename = "$value")]
        item: [Choice; 3],
    }

    #[test]
    fn before() {
        let data: Root = from_str(
            r#"
                    <root>
                        <node/>
                        <one/>
                        <two/>
                        <three/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                node: (),
                item: [Choice::One, Choice::Two, Choice::Other("three".into())],
            }
        );
    }

    #[test]
    fn after() {
        let data: Root = from_str(
            r#"
                    <root>
                        <one/>
                        <two/>
                        <three/>
                        <node/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                node: (),
                item: [Choice::One, Choice::Two, Choice::Other("three".into())],
            }
        );
    }

    #[test]
    fn overlapped() {
        let data = from_str::<Root>(
            r#"
                    <root>
                        <one/>
                        <node/>
                        <two/>
                        <three/>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                node: (),
                item: [Choice::One, Choice::Two, Choice::Other("three".into())],
            }
        );

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
            #[serde(rename = "$value")]
            item: [Choice4; 3],
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <one><inner/></one>
                        <node/>
                        <two><inner/></two>
                        <three><inner/></three>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                node: (),
                item: [
                    Choice4::One { inner: [()] },
                    Choice4::Two { inner: [()] },
                    Choice4::Other,
                ],
            }
        );

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
/// That struct should be deserialized from the XML where these
/// fields comes in an arbitrary order
mod field_after_list {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize)]
    struct Root {
        #[serde(rename = "$value")]
        item: [Choice; 3],
        node: (),
    }

    #[test]
    fn before() {
        let data: Root = from_str(
            r#"
                    <root>
                        <node/>
                        <one/>
                        <two/>
                        <three/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                node: (),
            }
        );
    }

    #[test]
    fn after() {
        let data: Root = from_str(
            r#"
                    <root>
                        <one/>
                        <two/>
                        <three/>
                        <node/>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(
            data,
            Root {
                item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                node: (),
            }
        );
    }

    #[test]
    fn overlapped() {
        let data = from_str::<Root>(
            r#"
                    <root>
                        <one/>
                        <node/>
                        <two/>
                        <three/>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                node: (),
            }
        );

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
            #[serde(rename = "$value")]
            item: [Choice4; 3],
            node: (),
        }

        let data = from_str::<Root>(
            r#"
                    <root>
                        <one><inner/></one>
                        <node/>
                        <two><inner/></two>
                        <three><inner/></three>
                    </root>
                    "#,
        );

        #[cfg(feature = "overlapped-lists")]
        assert_eq!(
            data.expect("should be Ok"),
            Root {
                item: [
                    Choice4::One { inner: [()] },
                    Choice4::Two { inner: [()] },
                    Choice4::Other,
                ],
                node: (),
            }
        );

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

    /// A field with a variable-name items defined before a field with a fixed-name
    /// items
    mod choice_and_fixed {
        use super::*;

        #[derive(Debug, PartialEq, Deserialize)]
        struct Pair {
            #[serde(rename = "$value")]
            item: [Choice; 3],
            element: [(); 2],
        }

        /// A list with fixed-name elements located before a list with variable-name
        /// elements in an XML
        #[test]
        fn fixed_before() {
            let data: Pair = from_str(
                r#"
                        <root>
                            <element/>
                            <element/>
                            <one/>
                            <two/>
                            <three/>
                        </root>
                        "#,
            )
            .expect("should be Ok");

            assert_eq!(
                data,
                Pair {
                    item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                    element: [(), ()],
                }
            );
        }

        /// A list with fixed-name elements located after a list with variable-name
        /// elements in an XML
        #[test]
        fn fixed_after() {
            let data: Pair = from_str(
                r#"
                        <root>
                            <one/>
                            <two/>
                            <three/>
                            <element/>
                            <element/>
                        </root>
                        "#,
            )
            .expect("should be Ok");

            assert_eq!(
                data,
                Pair {
                    item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                    element: [(), ()],
                }
            );
        }

        mod overlapped {
            use super::*;

            #[derive(Debug, PartialEq, Deserialize)]
            struct Root {
                #[serde(rename = "$value")]
                item: [Choice4; 3],
                element: [(); 2],
            }

            /// A list with fixed-name elements are mixed with a list with variable-name
            /// elements in an XML, and the first element is a fixed-name one
            #[test]
            fn fixed_before() {
                let data = from_str::<Pair>(
                    r#"
                            <root>
                                <element/>
                                <one/>
                                <two/>
                                <element/>
                                <three/>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Pair {
                        item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                        element: [(), ()],
                    }
                );

                #[cfg(not(feature = "overlapped-lists"))]
                match data {
                    Err(DeError::Custom(e)) => {
                        assert_eq!(e, "invalid length 1, expected an array of length 2")
                    }
                    e => panic!(
                        r#"Expected `Err(Custom("invalid length 1, expected an array of length 2"))`, but got `{:?}`"#,
                        e
                    ),
                }
            }

            /// A list with fixed-name elements are mixed with a list with variable-name
            /// elements in an XML, and the first element is a variable-name one
            #[test]
            fn fixed_after() {
                let data = from_str::<Pair>(
                    r#"
                            <root>
                                <one/>
                                <element/>
                                <two/>
                                <three/>
                                <element/>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Pair {
                        item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                        element: [(), ()],
                    }
                );

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
            fn with_nested_list_fixed_before() {
                let data = from_str::<Root>(
                    r#"
                            <root>
                                <element/>
                                <one><inner/></one>
                                <two><inner/></two>
                                <element/>
                                <three><inner/></three>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Root {
                        item: [
                            Choice4::One { inner: [()] },
                            Choice4::Two { inner: [()] },
                            Choice4::Other,
                        ],
                        element: [(); 2],
                    }
                );

                #[cfg(not(feature = "overlapped-lists"))]
                match data {
                    Err(DeError::Custom(e)) => {
                        assert_eq!(e, "invalid length 1, expected an array of length 2")
                    }
                    e => panic!(
                        r#"Expected `Err(Custom("invalid length 1, expected an array of length 2"))`, but got `{:?}`"#,
                        e
                    ),
                }
            }

            /// Test for https://github.com/tafia/quick-xml/issues/435
            #[test]
            fn with_nested_list_fixed_after() {
                let data = from_str::<Root>(
                    r#"
                            <root>
                                <one><inner/></one>
                                <element/>
                                <two><inner/></two>
                                <three><inner/></three>
                                <element/>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Root {
                        item: [
                            Choice4::One { inner: [()] },
                            Choice4::Two { inner: [()] },
                            Choice4::Other,
                        ],
                        element: [(); 2],
                    }
                );

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
    }

    /// A field with a variable-name items defined after a field with a fixed-name
    /// items
    mod fixed_and_choice {
        use super::*;

        #[derive(Debug, PartialEq, Deserialize)]
        struct Pair {
            element: [(); 2],
            #[serde(rename = "$value")]
            item: [Choice; 3],
        }

        /// A list with fixed-name elements located before a list with variable-name
        /// elements in an XML
        #[test]
        fn fixed_before() {
            let data: Pair = from_str(
                r#"
                        <root>
                            <element/>
                            <element/>
                            <one/>
                            <two/>
                            <three/>
                        </root>
                        "#,
            )
            .expect("should be Ok");

            assert_eq!(
                data,
                Pair {
                    item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                    element: [(), ()],
                }
            );
        }

        /// A list with fixed-name elements located after a list with variable-name
        /// elements in an XML
        #[test]
        fn fixed_after() {
            let data: Pair = from_str(
                r#"
                        <root>
                            <one/>
                            <two/>
                            <three/>
                            <element/>
                            <element/>
                        </root>
                        "#,
            )
            .expect("should be Ok");

            assert_eq!(
                data,
                Pair {
                    item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                    element: [(), ()],
                }
            );
        }

        mod overlapped {
            use super::*;

            #[derive(Debug, PartialEq, Deserialize)]
            struct Root {
                element: [(); 2],
                #[serde(rename = "$value")]
                item: [Choice4; 3],
            }

            /// A list with fixed-name elements are mixed with a list with variable-name
            /// elements in an XML, and the first element is a fixed-name one
            #[test]
            fn fixed_before() {
                let data = from_str::<Pair>(
                    r#"
                            <root>
                                <element/>
                                <one/>
                                <two/>
                                <element/>
                                <three/>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Pair {
                        item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                        element: [(), ()],
                    }
                );

                #[cfg(not(feature = "overlapped-lists"))]
                match data {
                    Err(DeError::Custom(e)) => {
                        assert_eq!(e, "invalid length 1, expected an array of length 2")
                    }
                    e => panic!(
                        r#"Expected `Err(Custom("invalid length 1, expected an array of length 2"))`, but got `{:?}`"#,
                        e
                    ),
                }
            }

            /// A list with fixed-name elements are mixed with a list with variable-name
            /// elements in an XML, and the first element is a variable-name one
            #[test]
            fn fixed_after() {
                let data = from_str::<Pair>(
                    r#"
                            <root>
                                <one/>
                                <element/>
                                <two/>
                                <three/>
                                <element/>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Pair {
                        item: [Choice::One, Choice::Two, Choice::Other("three".into())],
                        element: [(), ()],
                    }
                );

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
            fn with_nested_list_fixed_before() {
                let data = from_str::<Root>(
                    r#"
                            <root>
                                <element/>
                                <one><inner/></one>
                                <two><inner/></two>
                                <element/>
                                <three><inner/></three>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Root {
                        element: [(); 2],
                        item: [
                            Choice4::One { inner: [()] },
                            Choice4::Two { inner: [()] },
                            Choice4::Other,
                        ],
                    }
                );

                #[cfg(not(feature = "overlapped-lists"))]
                match data {
                    Err(DeError::Custom(e)) => {
                        assert_eq!(e, "invalid length 1, expected an array of length 2")
                    }
                    e => panic!(
                        r#"Expected `Err(Custom("invalid length 1, expected an array of length 2"))`, but got `{:?}`"#,
                        e
                    ),
                }
            }

            /// Test for https://github.com/tafia/quick-xml/issues/435
            #[test]
            fn with_nested_list_fixed_after() {
                let data = from_str::<Root>(
                    r#"
                            <root>
                                <one><inner/></one>
                                <element/>
                                <two><inner/></two>
                                <three><inner/></three>
                                <element/>
                            </root>
                            "#,
                );

                #[cfg(feature = "overlapped-lists")]
                assert_eq!(
                    data.expect("should be Ok"),
                    Root {
                        element: [(); 2],
                        item: [
                            Choice4::One { inner: [()] },
                            Choice4::Two { inner: [()] },
                            Choice4::Other,
                        ],
                    }
                );

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
    }

    // choice_and_choice struct like
    //
    //      #[derive(Deserialize)]
    //      struct Pair {
    //          #[serde(rename = "$value")]
    //          item: [Choice; 2],
    //
    //          #[serde(rename = "$value")]
    //          element: [Choice2; 3],
    //      }
    //
    // cannot be implemented because both fields should be renamed to the
    // same name $value. Derived implementation will never try to deserialize
    // `element`
}

/// Deserialization of primitives slightly differs from deserialization
/// of complex types, so need to check this separately
#[test]
fn primitives() {
    #[derive(Debug, PartialEq, Deserialize)]
    struct List {
        #[serde(rename = "$value")]
        item: [Choice3; 3],
    }

    let data: List = from_str(
        r#"
                <root>
                    <one>41</one>
                    <two>42</two>
                    <three>43</three>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: [
                Choice3::One(41),
                Choice3::Two("42".to_string()),
                Choice3::Other,
            ],
        }
    );

    from_str::<List>(
        r#"
                <root>
                    <one>41</one>
                    <two><item>42</item></two>
                    <three>43</three>
                </root>
                "#,
    )
    .expect_err("should be Err");
}

/// Test for https://github.com/tafia/quick-xml/issues/567
#[test]
fn list_of_enum() {
    #[derive(Debug, PartialEq, Deserialize)]
    enum Enum {
        Variant(Vec<String>),
    }

    let data: List<Enum> = from_str(
        r#"
                <root>
                    <Variant>first item</Variant>
                    <Variant>second item</Variant>
                    <Variant>third item</Variant>
                </root>
                "#,
    )
    .expect("should be Ok");

    assert_eq!(
        data,
        List {
            item: [
                Enum::Variant(vec!["first".to_string(), "item".to_string()]),
                Enum::Variant(vec!["second".to_string(), "item".to_string()]),
                Enum::Variant(vec!["third".to_string(), "item".to_string()]),
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
        #[serde(rename = "$value")]
        value: [(); 2],
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
/// represented by `xs:list`s
mod xs_list {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct List {
        /// Outer list mapped to elements, inner -- to `xs:list`.
        ///
        /// `#[serde(default)]` is not required, because correct
        /// XML will always contains at least 1 element.
        #[serde(rename = "$value")]
        element: [Vec<String>; 1],
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
            #[serde(rename = "$value")]
            element: [Vec<String>; 0],
        }

        let data: List = from_str(
            r#"
                    <root>
                    </root>
                    "#,
        )
        .expect("should be Ok");

        assert_eq!(data, List { element: [] });
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
                element: [vec!["first".to_string(), "list".to_string()]]
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

        assert_eq!(data, List { element: [vec![]] });
    }

    /// Special case: outer list is always mapped to an elements sequence,
    /// not to an `xs:list`
    #[test]
    fn element() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct List {
            /// List mapped to elements, String -- to `xs:list`.
            ///
            /// `#[serde(default)]` is not required, because correct
            /// XML will always contains at least 1 element.
            #[serde(rename = "$value")]
            element: [String; 1],
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
                element: ["first item".to_string()]
            }
        );
    }

    #[test]
    fn many() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct List {
            /// Outer list mapped to elements, inner -- to `xs:list`
            #[serde(rename = "$value")]
            element: [Vec<String>; 2],
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
                element: [
                    vec!["first".to_string(), "list".to_string()],
                    vec!["second".to_string(), "list".to_string()],
                ]
            }
        );
    }
}
