//! `serde` support for [`Iri`] and [`IriRef`] (the `serde` feature).
//!
//! Both types serialize as their plain string form and deserialize via `parse` (a parse
//! failure becomes a `serde::de::Error::custom`) — exactly the shape upstream `oxiri`'s own
//! opt-in `serde` feature uses (`Serialize for IriRef<T>` forwards straight to `T`'s own
//! `Serialize`, with no wrapping object; `Deserialize` re-validates through `parse`/`TryFrom`).
//!
//! The workspace pulls in no `serde` data format (no `serde_json`, no `serde_test`), so this
//! file carries a minimal, self-contained string-shaped [`Serializer`] together with
//! `serde::de::value`'s [`StrDeserializer`]/[`BorrowedStrDeserializer`] adapters, which is all
//! that is needed: `Serialize for IriRef<T>`/`Iri<T>` only ever issues a single `serialize_str`
//! call, and `Deserialize` only ever asks its deserializer for a string.

#![cfg(feature = "serde")]

use std::fmt;

use oxixml_iri::{Iri, IriRef};
use serde::de::value::{BorrowedStrDeserializer, Error as ValueError, StrDeserializer};
use serde::ser::Impossible;
use serde::{Deserialize, Serialize, Serializer};

type TestResult = Result<(), Box<dyn std::error::Error>>;

// -------------------------------------------------------------------------------------------
// A minimal string-only `Serializer`: `Iri`/`IriRef` only ever call `serialize_str` (both
// `str` and `String`'s own `Serialize` impls forward to it), so every other method is
// unreachable in practice and simply reports itself as unsupported.
// -------------------------------------------------------------------------------------------

#[derive(Debug)]
struct CaptureError(String);

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CaptureError {}

impl serde::ser::Error for CaptureError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

/// A [`Serializer`] that only ever accepts a bare string.
struct StringSerializer;

macro_rules! unsupported {
    ($($name:ident($($arg:ty),*);)*) => {
        $(
            fn $name(self $(, _: $arg)*) -> Result<String, CaptureError> {
                Err(CaptureError("unsupported by the test serializer".to_owned()))
            }
        )*
    };
}

impl Serializer for StringSerializer {
    type Ok = String;
    type Error = CaptureError;
    type SerializeSeq = Impossible<String, CaptureError>;
    type SerializeTuple = Impossible<String, CaptureError>;
    type SerializeTupleStruct = Impossible<String, CaptureError>;
    type SerializeTupleVariant = Impossible<String, CaptureError>;
    type SerializeMap = Impossible<String, CaptureError>;
    type SerializeStruct = Impossible<String, CaptureError>;
    type SerializeStructVariant = Impossible<String, CaptureError>;

    fn serialize_str(self, value: &str) -> Result<String, CaptureError> {
        Ok(value.to_owned())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<String, CaptureError> {
        value.serialize(self)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<String, CaptureError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<String, CaptureError> {
        Err(CaptureError(
            "unsupported by the test serializer".to_owned(),
        ))
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, CaptureError> {
        Err(CaptureError("sequences are unsupported".to_owned()))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, CaptureError> {
        Err(CaptureError("tuples are unsupported".to_owned()))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, CaptureError> {
        Err(CaptureError("tuple structs are unsupported".to_owned()))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, CaptureError> {
        Err(CaptureError("tuple variants are unsupported".to_owned()))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, CaptureError> {
        Err(CaptureError("maps are unsupported".to_owned()))
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, CaptureError> {
        Err(CaptureError("structs are unsupported".to_owned()))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, CaptureError> {
        Err(CaptureError("struct variants are unsupported".to_owned()))
    }

    unsupported! {
        serialize_bool(bool);
        serialize_i8(i8);
        serialize_i16(i16);
        serialize_i32(i32);
        serialize_i64(i64);
        serialize_u8(u8);
        serialize_u16(u16);
        serialize_u32(u32);
        serialize_u64(u64);
        serialize_f32(f32);
        serialize_f64(f64);
        serialize_char(char);
        serialize_bytes(&[u8]);
        serialize_none();
        serialize_unit();
        serialize_unit_struct(&'static str);
        serialize_unit_variant(&'static str, u32, &'static str);
    }
}

// -------------------------------------------------------------------------------------------
// `IriRef` tests.
// -------------------------------------------------------------------------------------------

#[test]
fn iri_ref_borrowed_serializes_to_its_plain_string_form() -> TestResult {
    let iri_ref: IriRef<&str> = IriRef::parse("//example.com/bar/baz")?;
    assert_eq!(
        iri_ref.serialize(StringSerializer)?,
        "//example.com/bar/baz"
    );
    Ok(())
}

#[test]
fn iri_ref_owned_serializes_to_its_plain_string_form() -> TestResult {
    let iri_ref: IriRef<String> = IriRef::parse("//example.com/bar/baz".to_owned())?;
    assert_eq!(
        iri_ref.serialize(StringSerializer)?,
        "//example.com/bar/baz"
    );
    Ok(())
}

#[test]
fn iri_ref_owned_round_trips_through_a_plain_string() -> TestResult {
    let original: IriRef<String> = IriRef::parse("../bar/baz?q#f".to_owned())?;
    let serialized = original.serialize(StringSerializer)?;
    assert_eq!(serialized, "../bar/baz?q#f");

    let deserializer: StrDeserializer<ValueError> = StrDeserializer::new(&serialized);
    let decoded = IriRef::<String>::deserialize(deserializer)?;
    assert_eq!(decoded, original);
    Ok(())
}

#[test]
fn iri_ref_deserializes_zero_copy_into_a_borrowed_str() -> TestResult {
    let text = "//example.com/bar/baz".to_owned();
    let deserializer: BorrowedStrDeserializer<ValueError> = BorrowedStrDeserializer::new(&text);
    let decoded = IriRef::<&str>::deserialize(deserializer)?;
    assert_eq!(decoded.as_str(), text);
    Ok(())
}

#[test]
fn iri_ref_deserialize_rejects_invalid_input_with_a_custom_error() {
    let deserializer: StrDeserializer<ValueError> = StrDeserializer::new(":");
    let error = IriRef::<String>::deserialize(deserializer).expect_err("must fail to parse");
    assert!(
        error.to_string().contains("empty schemes are not allowed"),
        "unexpected error message: {error}"
    );
}

// -------------------------------------------------------------------------------------------
// `Iri` tests.
// -------------------------------------------------------------------------------------------

#[test]
fn iri_borrowed_serializes_to_its_plain_string_form() -> TestResult {
    let iri: Iri<&str> = Iri::parse("http://example.com/bar/baz")?;
    assert_eq!(
        iri.serialize(StringSerializer)?,
        "http://example.com/bar/baz"
    );
    Ok(())
}

#[test]
fn iri_owned_serializes_to_its_plain_string_form() -> TestResult {
    let iri: Iri<String> = Iri::parse("http://example.com/bar/baz".to_owned())?;
    assert_eq!(
        iri.serialize(StringSerializer)?,
        "http://example.com/bar/baz"
    );
    Ok(())
}

#[test]
fn iri_owned_round_trips_through_a_plain_string() -> TestResult {
    let original: Iri<String> = Iri::parse("http://example.com/bar/baz?q#f".to_owned())?;
    let serialized = original.serialize(StringSerializer)?;
    assert_eq!(serialized, "http://example.com/bar/baz?q#f");

    let deserializer: StrDeserializer<ValueError> = StrDeserializer::new(&serialized);
    let decoded = Iri::<String>::deserialize(deserializer)?;
    assert_eq!(decoded, original);
    Ok(())
}

#[test]
fn iri_deserializes_zero_copy_into_a_borrowed_str() -> TestResult {
    let text = "http://example.com/bar/baz".to_owned();
    let deserializer: BorrowedStrDeserializer<ValueError> = BorrowedStrDeserializer::new(&text);
    let decoded = Iri::<&str>::deserialize(deserializer)?;
    assert_eq!(decoded.as_str(), text);
    Ok(())
}

#[test]
fn iri_deserialize_rejects_a_malformed_string_with_a_custom_error() {
    let deserializer: StrDeserializer<ValueError> = StrDeserializer::new(":");
    let error = Iri::<String>::deserialize(deserializer).expect_err("must fail to parse");
    assert!(
        error.to_string().contains("empty schemes are not allowed"),
        "unexpected error message: {error}"
    );
}

#[test]
fn iri_deserialize_rejects_a_relative_reference_with_a_custom_error() {
    // Valid as an `IriRef` (an authority-only relative reference), but not absolute -- `Iri`'s
    // `Deserialize` must reject it via the `TryFrom<IriRef<T>>` "no scheme" path, not silently
    // accept whatever `IriRef::parse` accepted.
    let deserializer: StrDeserializer<ValueError> = StrDeserializer::new("//example.com");
    let error =
        Iri::<String>::deserialize(deserializer).expect_err("must fail: not an absolute IRI");
    assert!(
        error
            .to_string()
            .contains("no scheme found in an absolute IRI"),
        "unexpected error message: {error}"
    );
}

#[test]
fn iri_and_iri_ref_serialize_identically_for_the_same_text() -> TestResult {
    // `Serialize for Iri<T>` forwards to its wrapped `IriRef<T>` -- confirm the two really do
    // produce the same wire value for the same underlying text.
    let iri: Iri<String> = Iri::parse("http://example.com/a".to_owned())?;
    let iri_ref: IriRef<String> = IriRef::parse("http://example.com/a".to_owned())?;
    assert_eq!(
        iri.serialize(StringSerializer)?,
        iri_ref.serialize(StringSerializer)?
    );
    Ok(())
}
