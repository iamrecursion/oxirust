#![doc = include_str!("overview.md")]
#![doc = include_str!("text-and-value.md")]

#[macro_use]
mod macros;

mod attributes;
mod deserializer;
mod event;
mod key;
mod map;
mod resolver;
mod simple_type;
mod text;
mod var;
mod xml_read;

#[cfg(test)]
mod tests;

pub use self::attributes::AttributesDeserializer;
pub use self::deserializer::{Deserializer, from_reader, from_str};
pub use self::event::{DeEvent, PayloadEvent, Text};
pub use self::resolver::{EntityResolver, PredefinedEntityResolver};
pub use self::simple_type::SimpleTypeDeserializer;
pub use self::xml_read::{IoReader, SliceReader, XmlRead};
pub use crate::errors::serialize::DeError;

/// Data represented by a text node or a CDATA node. XML markup is not expected
pub(crate) const TEXT_KEY: &str = "$text";
/// Data represented by any XML markup inside
pub(crate) const VALUE_KEY: &str = "$value";
