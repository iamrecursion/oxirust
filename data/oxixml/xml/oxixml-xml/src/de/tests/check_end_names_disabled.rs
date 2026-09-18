//! [`Config::check_end_names`] is documented to permit `<mytag></different_tag>`
//! when disabled. `ElementMapAccess::next_key_seed` and
//! `MapValueSeqAccess::next_element_seed` both terminate on any `End` event once
//! their owning tag has started, but used to first run a `debug_assert_eq!` that
//! re-checked the very fact the config flag waives — a check that only ever
//! fires in a dev/test build (`cargo test` runs with `debug_assertions` on),
//! exactly the build profile the option exists to support.
//!
//! [`Config::check_end_names`]: crate::reader::Config::check_end_names

use super::*;
use serde::Deserialize;

fn lenient_de(xml: &str) -> Deserializer<'_, SliceReader<'_>> {
    let mut reader = NsReader::from_str(xml);
    reader.config_mut().check_end_names = false;
    Deserializer::borrowing(reader)
}

#[test]
fn struct_field_map_accepts_mismatched_end_name() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct Foo {
        a: String,
    }

    let mut de = lenient_de("<Foo><a>1</a></Bar>");
    assert_eq!(
        Foo::deserialize(&mut de).expect("should be Ok"),
        Foo { a: "1".into() }
    );
}

#[test]
fn sequence_field_accepts_mismatched_end_name() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct FooSeq {
        item: Vec<String>,
    }

    let mut de = lenient_de("<Foo><item>a</item><item>b</item></Bar>");
    assert_eq!(
        FooSeq::deserialize(&mut de).expect("should be Ok"),
        FooSeq {
            item: vec!["a".into(), "b".into()]
        }
    );
}
