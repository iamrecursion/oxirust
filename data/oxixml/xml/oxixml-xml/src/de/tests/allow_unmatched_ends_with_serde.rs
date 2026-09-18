//! [`Config::allow_unmatched_ends`] lets the reader emit an [`Event::End`] with
//! no corresponding open tag anywhere in the document. Every "fresh value" entry
//! point of the deserializer used to assume that could never happen and panicked
//! via `unreachable!()`; each one must now return an ordinary `Err` instead.
//!
//! [`Config::allow_unmatched_ends`]: crate::reader::Config::allow_unmatched_ends
//! [`Event::End`]: crate::events::Event::End

use super::*;
use serde::Deserialize;

/// Builds a `Deserializer` over `xml` with `allow_unmatched_ends` turned on,
/// exactly the public constructor pattern this crate's own rustdoc demonstrates.
fn permissive_de(xml: &str) -> Deserializer<'_, SliceReader<'_>> {
    let mut reader = NsReader::from_str(xml);
    reader.config_mut().allow_unmatched_ends = true;
    Deserializer::borrowing(reader)
}

#[test]
fn primitive() {
    let mut de = permissive_de("</b>");
    match String::deserialize(&mut de) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("b".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(UnmatchedEndTag(_))))`, but got `{:?}`",
            x
        ),
    }
}

#[test]
fn struct_() {
    #[derive(Debug, Deserialize)]
    #[allow(unused)]
    struct Foo {
        a: String,
    }

    let mut de = permissive_de("</b>");
    match Foo::deserialize(&mut de) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("b".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(UnmatchedEndTag(_))))`, but got `{:?}`",
            x
        ),
    }
}

#[test]
fn unit_struct() {
    #[derive(Debug, Deserialize)]
    struct Unit;

    let mut de = permissive_de("</b>");
    match Unit::deserialize(&mut de) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("b".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(UnmatchedEndTag(_))))`, but got `{:?}`",
            x
        ),
    }
}

#[test]
fn enum_() {
    #[derive(Debug, Deserialize)]
    #[allow(unused)]
    enum E {
        A,
        B,
    }

    let mut de = permissive_de("</b>");
    match E::deserialize(&mut de) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("b".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(UnmatchedEndTag(_))))`, but got `{:?}`",
            x
        ),
    }
}

/// A leading stray end followed by a well-formed element: the deserializer
/// must reject the stray `</b>` up front rather than skip past it.
#[test]
fn leading_stray_end_before_valid_document() {
    #[derive(Debug, Deserialize)]
    #[allow(unused)]
    struct Foo {
        a: String,
    }

    let mut de = permissive_de("</b><Foo><a>1</a></Foo>");
    match Foo::deserialize(&mut de) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("b".into()))
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(UnmatchedEndTag(_))))`, but got `{:?}`",
            x
        ),
    }
}
