//! Unit tests for the deserializer event pipeline.

use super::*;
use crate::XmlVersion;
use crate::de::event::PayloadEvent;
use crate::errors::{Error, IllFormedError};
use crate::events::{BytesEnd, BytesStart, BytesText};
use crate::name::QName;
use crate::reader::NsReader;

fn make_de<'de>(source: &'de str) -> Deserializer<'de, SliceReader<'de>> {
    dbg!(source);
    Deserializer::from_str(source)
}

mod allow_unmatched_ends_with_serde;
mod check_end_names_disabled;
mod merge_text;
mod misc;
mod read_to_end;
#[cfg(feature = "overlapped-lists")]
mod skip;
mod triples;
