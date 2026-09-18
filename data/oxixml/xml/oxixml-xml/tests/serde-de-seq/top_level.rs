use super::*;

#[test]
fn simple() {
    from_str::<[(); 3]>("<root/><root>42</root><root>answer</root>").expect("should be Ok");

    let data: Vec<()> =
        from_str("<root/><root>42</root><root>answer</root>").expect("should be Ok");
    assert_eq!(data, vec![(), (), ()]);
}

/// Special case: empty sequence
#[test]
fn empty() {
    from_str::<[(); 0]>("").expect("should be Ok");

    let data: Vec<()> = from_str("").expect("should be Ok");
    assert_eq!(data, vec![]);
}

/// Special case: one-element sequence
#[test]
fn one_element() {
    from_str::<[(); 1]>("<root/>").expect("should be Ok");
    from_str::<[(); 1]>("<root>42</root>").expect("should be Ok");
    from_str::<[(); 1]>("text").expect("should be Ok");
    from_str::<[(); 1]>("<![CDATA[cdata]]>").expect("should be Ok");

    let data: Vec<()> = from_str("<root/>").expect("should be Ok");
    assert_eq!(data, vec![()]);

    let data: Vec<()> = from_str("<root>42</root>").expect("should be Ok");
    assert_eq!(data, vec![()]);

    let data: Vec<()> = from_str("text").expect("should be Ok");
    assert_eq!(data, vec![()]);

    let data: Vec<()> = from_str("<![CDATA[cdata]]>").expect("should be Ok");
    assert_eq!(data, vec![()]);
}

#[test]
fn excess_attribute() {
    from_str::<[(); 3]>(r#"<root/><root excess="attribute">42</root><root>answer</root>"#)
        .expect("should be Ok");

    let data: Vec<()> = from_str(r#"<root/><root excess="attribute">42</root><root>answer</root>"#)
        .expect("should be Ok");
    assert_eq!(data, vec![(), (), ()]);
}

#[test]
fn mixed_content() {
    // Text and CDATA represents a one logical text item
    from_str::<[(); 2]>(
        r#"
            <element/>
            text
            <![CDATA[cdata]]>
            "#,
    )
    .expect("should be Ok");

    let data: Vec<()> = from_str(
        r#"
            <element/>
            text
            <![CDATA[cdata]]>
            "#,
    )
    .expect("should be Ok");
    // Text and CDATA represents a one logical text item
    assert_eq!(data, vec![(), ()]);
}

/// This test ensures that composition of deserializer building blocks plays well
#[test]
fn list_of_struct() {
    #[derive(Debug, PartialEq, Default, Deserialize)]
    #[serde(default)]
    struct Struct {
        #[serde(rename = "@attribute")]
        attribute: Option<String>,
        element: Option<String>,
    }

    let data: Vec<Struct> = from_str(
        r#"
            <struct/>
            <struct attribute="value"/>
            <struct>
                <element>value</element>
            </struct>
            <struct attribute="value">
                <element>value</element>
            </struct>
            "#,
    )
    .expect("should be Ok");
    assert_eq!(
        data,
        vec![
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
        ]
    );
}

/// Test for https://github.com/tafia/quick-xml/issues/500
#[test]
fn list_of_enum() {
    #[derive(Debug, PartialEq, Deserialize)]
    enum Enum {
        One,
        Two,
    }

    let data: Vec<Enum> = from_str(
        r#"
            <One/>
            <Two/>
            <One/>
            "#,
    )
    .expect("should be Ok");
    assert_eq!(data, vec![Enum::One, Enum::Two, Enum::One]);
}
