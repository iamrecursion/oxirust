//! Atomic values: the [`AtomicType`] hierarchy, the [`AtomicValue`] carrier, the lexical
//! mapping, and the comparison relations XPath and XQuery are defined in terms of.

mod bignum;
mod binary;
mod compare;
mod lexical;
mod parse;
mod qname_value;
mod types;
mod value;

pub use bignum::{BigDecimal, BigInteger};

pub use compare::{
    CODEPOINT_COLLATION, HTML_ASCII_CASE_INSENSITIVE_COLLATION, atomic_equal,
    atomic_equal_in_context, atomic_equal_with_collation, compare_atomic,
    compare_atomic_in_context, compare_atomic_with_collation, compare_strings, same_key,
    same_key_hash,
};
pub use qname_value::QNameValue;
pub use types::{AtomicType, XSD_NAMESPACE};
pub use value::{AtomicData, AtomicValue};
