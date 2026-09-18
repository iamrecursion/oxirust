//! Tests for [`Deserializer::read_to_end()`].

use super::*;
use crate::de::DeEvent::*;

#[test]
fn complex() {
    let mut de = make_de(
        r#"
                <root>
                    <tag a="1"><tag>text</tag>content</tag>
                    <tag a="2"><![CDATA[cdata content]]></tag>
                    <self-closed/>
                </root>
                "#,
    );

    assert_eq!(
        de.next().expect("should be Ok"),
        Text("\n                ".into())
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("root"))
    );

    assert_eq!(
        de.next().expect("should be Ok"),
        Text("\n                    ".into())
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::from_content(r#"tag a="1""#, 3))
    );
    assert_eq!(de.read_to_end(QName(b"tag")).expect("should be Ok"), ());

    assert_eq!(
        de.next().expect("should be Ok"),
        Text("\n                    ".into())
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::from_content(r#"tag a="2""#, 3))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Text("cdata content".into())
    );
    assert_eq!(de.next().expect("should be Ok"), End(BytesEnd::new("tag")));

    assert_eq!(
        de.next().expect("should be Ok"),
        Text("\n                    ".into())
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("self-closed"))
    );
    assert_eq!(
        de.read_to_end(QName(b"self-closed")).expect("should be Ok"),
        ()
    );

    assert_eq!(
        de.next().expect("should be Ok"),
        Text("\n                ".into())
    );
    assert_eq!(de.next().expect("should be Ok"), End(BytesEnd::new("root")));
    assert_eq!(
        de.next().expect("should be Ok"),
        Text("\n                ".into())
    );
    assert_eq!(de.next().expect("should be Ok"), Eof);
}

#[test]
fn invalid_xml1() {
    let mut de = make_de("<tag><tag></tag>");

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("tag"))
    );
    assert_eq!(
        de.peek().expect("should be Ok"),
        &Start(BytesStart::new("tag"))
    );

    match de.read_to_end(QName(b"tag")) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::MissingEndTag("tag".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
            x
        ),
    }
    assert_eq!(de.next().expect("should be Ok"), Eof);
}

#[test]
fn invalid_xml2() {
    let mut de = make_de("<tag><![CDATA[]]><tag></tag>");

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("tag"))
    );
    assert_eq!(de.peek().expect("should be Ok"), &Text("".into()));

    match de.read_to_end(QName(b"tag")) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::MissingEndTag("tag".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
            x
        ),
    }
    assert_eq!(de.next().expect("should be Ok"), Eof);
}
