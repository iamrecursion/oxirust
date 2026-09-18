//! `serde` support for the RDF term and statement types (the `serde` feature).
//!
//! The workspace pulls in no `serde` data format, so this file carries a minimal, self-contained
//! JSON-shaped [`Value`] model together with the [`serde::Serializer`] and
//! [`serde::Deserializer`] adapters needed to exercise the implementations end to end.

#![cfg(feature = "serde")]

use oxixml_model::{
    BlankNode, GraphName, Literal, NamedNode, NamedOrBlankNode, OxString, Quad, Term, Triple,
};
use serde::de::value::{Error as ValueError, MapDeserializer, SeqDeserializer};
use serde::de::{DeserializeOwned, IntoDeserializer, Visitor};
use serde::ser::{Impossible, SerializeMap, SerializeSeq, SerializeStruct};
use serde::{Deserializer, Serialize, Serializer, forward_to_deserialize_any};
use std::fmt;

type TestResult = Result<(), Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------------------------
// A tiny self-describing value model, plus the serde adapters around it.
// ---------------------------------------------------------------------------------------------

/// A minimal JSON-shaped value: everything the RDF model's `serde` impls can produce.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    Unit,
    Bool(bool),
    Str(String),
    Seq(Vec<Value>),
    Map(Vec<(String, Value)>),
}

impl Value {
    /// Renders the value as canonical JSON, so tests can assert the exact wire shape.
    fn to_json(&self) -> String {
        match self {
            Value::Unit => "null".to_owned(),
            Value::Bool(value) => value.to_string(),
            Value::Str(value) => format!("{value:?}"),
            Value::Seq(items) => {
                let items: Vec<String> = items.iter().map(Value::to_json).collect();
                format!("[{}]", items.join(","))
            }
            Value::Map(entries) => {
                let entries: Vec<String> = entries
                    .iter()
                    .map(|(key, value)| format!("{:?}:{}", key, value.to_json()))
                    .collect();
                format!("{{{}}}", entries.join(","))
            }
        }
    }
}

#[derive(Debug)]
struct SerError(String);

impl fmt::Display for SerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SerError {}

impl serde::ser::Error for SerError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

/// Serializes any [`Serialize`] value into the [`Value`] model above.
struct ValueSerializer;

macro_rules! unsupported {
    ($($name:ident($($arg:ty),*);)*) => {
        $(
            fn $name(self $(, _: $arg)*) -> Result<Value, SerError> {
                Err(SerError("unsupported by the test serializer".to_owned()))
            }
        )*
    };
}

impl Serializer for ValueSerializer {
    type Ok = Value;
    type Error = SerError;
    type SerializeSeq = SeqBuilder;
    type SerializeTuple = Impossible<Value, SerError>;
    type SerializeTupleStruct = Impossible<Value, SerError>;
    type SerializeTupleVariant = Impossible<Value, SerError>;
    type SerializeMap = MapBuilder;
    type SerializeStruct = MapBuilder;
    type SerializeStructVariant = Impossible<Value, SerError>;

    fn serialize_bool(self, value: bool) -> Result<Value, SerError> {
        Ok(Value::Bool(value))
    }

    fn serialize_str(self, value: &str) -> Result<Value, SerError> {
        Ok(Value::Str(value.to_owned()))
    }

    fn serialize_none(self) -> Result<Value, SerError> {
        Ok(Value::Unit)
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Value, SerError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Value, SerError> {
        Ok(Value::Unit)
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<Value, SerError> {
        Ok(Value::Unit)
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Value, SerError> {
        Ok(Value::Str(variant.to_owned()))
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<Value, SerError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value, SerError> {
        Ok(Value::Map(vec![(
            variant.to_owned(),
            value.serialize(self)?,
        )]))
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<SeqBuilder, SerError> {
        Ok(SeqBuilder(Vec::new()))
    }

    fn serialize_map(self, _: Option<usize>) -> Result<MapBuilder, SerError> {
        Ok(MapBuilder {
            entries: Vec::new(),
            key: None,
        })
    }

    fn serialize_struct(self, _: &'static str, _: usize) -> Result<MapBuilder, SerError> {
        self.serialize_map(None)
    }

    fn serialize_tuple(self, _: usize) -> Result<Impossible<Value, SerError>, SerError> {
        Err(SerError("tuples are unsupported".to_owned()))
    }

    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Impossible<Value, SerError>, SerError> {
        Err(SerError("tuple structs are unsupported".to_owned()))
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Impossible<Value, SerError>, SerError> {
        Err(SerError("tuple variants are unsupported".to_owned()))
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Impossible<Value, SerError>, SerError> {
        Err(SerError("struct variants are unsupported".to_owned()))
    }

    unsupported! {
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
    }
}

struct SeqBuilder(Vec<Value>);

impl SerializeSeq for SeqBuilder {
    type Ok = Value;
    type Error = SerError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), SerError> {
        self.0.push(value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<Value, SerError> {
        Ok(Value::Seq(self.0))
    }
}

struct MapBuilder {
    entries: Vec<(String, Value)>,
    key: Option<String>,
}

impl SerializeMap for MapBuilder {
    type Ok = Value;
    type Error = SerError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), SerError> {
        match key.serialize(ValueSerializer)? {
            Value::Str(key) => {
                self.key = Some(key);
                Ok(())
            }
            other => Err(SerError(format!("map keys must be strings, got {other:?}"))),
        }
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), SerError> {
        let key = self
            .key
            .take()
            .ok_or_else(|| SerError("a map value without a key".to_owned()))?;
        self.entries.push((key, value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn end(self) -> Result<Value, SerError> {
        Ok(Value::Map(self.entries))
    }
}

impl SerializeStruct for MapBuilder {
    type Ok = Value;
    type Error = SerError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), SerError> {
        self.entries
            .push((key.to_owned(), value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn end(self) -> Result<Value, SerError> {
        Ok(Value::Map(self.entries))
    }
}

impl<'de> IntoDeserializer<'de, ValueError> for Value {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}

impl<'de> Deserializer<'de> for Value {
    type Error = ValueError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self {
            Value::Unit => visitor.visit_unit(),
            Value::Bool(value) => visitor.visit_bool(value),
            Value::Str(value) => visitor.visit_string(value),
            Value::Seq(items) => visitor.visit_seq(SeqDeserializer::new(items.into_iter())),
            Value::Map(entries) => visitor.visit_map(MapDeserializer::new(entries.into_iter())),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self {
            Value::Unit => visitor.visit_none(),
            other => visitor.visit_some(other),
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        match self {
            Value::Str(variant) => visitor.visit_enum(variant.into_deserializer()),
            other => other.deserialize_any(visitor),
        }
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
        unit unit_struct newtype_struct seq tuple tuple_struct map struct identifier
        ignored_any
    }
}

/// Serializes `value`, asserting the exact JSON shape, then reads it back.
fn round_trip<T>(value: &T, expected_json: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: Serialize + DeserializeOwned + PartialEq + fmt::Debug,
{
    let serialized = value.serialize(ValueSerializer)?;
    assert_eq!(serialized.to_json(), expected_json);
    let decoded = T::deserialize(serialized)?;
    assert_eq!(&decoded, value);
    Ok(decoded)
}

// ---------------------------------------------------------------------------------------------
// The tests themselves.
// ---------------------------------------------------------------------------------------------

#[test]
fn ox_string_is_a_plain_string() -> TestResult {
    round_trip(&OxString::new_owned("hello"), "\"hello\"")?;
    Ok(())
}

#[test]
fn named_node_round_trips() -> TestResult {
    round_trip(
        &NamedNode::new("http://example.com/foo")?,
        r#"{"value":"http://example.com/foo"}"#,
    )?;
    Ok(())
}

#[test]
fn blank_node_round_trips() -> TestResult {
    round_trip(&BlankNode::new("b0")?, r#"{"value":"b0"}"#)?;
    Ok(())
}

#[test]
fn literals_round_trip_in_every_shape() -> TestResult {
    round_trip(&Literal::new_simple_literal("foo"), r#"{"value":"foo"}"#)?;
    round_trip(
        &Literal::new_language_tagged_literal("foo", "en")?,
        r#"{"value":"foo","language":"en"}"#,
    )?;
    round_trip(
        &Literal::new_typed_literal("42", NamedNode::new("http://example.com/d")?),
        r#"{"value":"42","datatype":"http://example.com/d"}"#,
    )?;
    Ok(())
}

#[test]
fn term_unions_carry_their_type_tag() -> TestResult {
    round_trip(
        &Term::from(NamedNode::new("http://example.com/o")?),
        r#"{"type":"uri","value":"http://example.com/o"}"#,
    )?;
    round_trip(
        &Term::from(BlankNode::new("b0")?),
        r#"{"type":"bnode","value":"b0"}"#,
    )?;
    round_trip(
        &Term::from(Literal::new_simple_literal("x")),
        r#"{"type":"literal","value":"x"}"#,
    )?;
    round_trip(
        &NamedOrBlankNode::from(NamedNode::new("http://example.com/s")?),
        r#"{"type":"uri","value":"http://example.com/s"}"#,
    )?;
    round_trip(&GraphName::DefaultGraph, r#"{"type":"default"}"#)?;
    round_trip(
        &GraphName::from(NamedNode::new("http://example.com/g")?),
        r#"{"type":"uri","value":"http://example.com/g"}"#,
    )?;
    Ok(())
}

#[test]
fn statements_round_trip() -> TestResult {
    let triple = Triple::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_simple_literal("o"),
    );
    round_trip(
        &triple,
        concat!(
            r#"{"subject":{"type":"uri","value":"http://example.com/s"},"#,
            r#""predicate":{"type":"uri","value":"http://example.com/p"},"#,
            r#""object":{"type":"literal","value":"o"}}"#
        ),
    )?;

    let quad = Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        NamedNode::new("http://example.com/o")?,
        GraphName::from(NamedNode::new("http://example.com/g")?),
    );
    round_trip(
        &quad,
        concat!(
            r#"{"subject":{"type":"uri","value":"http://example.com/s"},"#,
            r#""predicate":{"type":"uri","value":"http://example.com/p"},"#,
            r#""object":{"type":"uri","value":"http://example.com/o"},"#,
            r#""graph":{"type":"uri","value":"http://example.com/g"}}"#
        ),
    )?;
    Ok(())
}

#[test]
fn borrowed_variants_serialize_like_their_owned_counterparts() -> TestResult {
    let node = NamedNode::new("http://example.com/s")?;
    assert_eq!(
        node.as_ref().serialize(ValueSerializer)?,
        node.serialize(ValueSerializer)?
    );
    let literal = Literal::new_language_tagged_literal("foo", "en")?;
    assert_eq!(
        literal.as_ref().serialize(ValueSerializer)?,
        literal.serialize(ValueSerializer)?
    );
    let quad = Quad::new(
        node.clone(),
        node.clone(),
        node.clone(),
        GraphName::DefaultGraph,
    );
    assert_eq!(
        quad.as_ref().serialize(ValueSerializer)?,
        quad.serialize(ValueSerializer)?
    );
    Ok(())
}

#[cfg(feature = "rdf-12")]
#[test]
fn rdf_12_terms_round_trip() -> TestResult {
    use oxixml_model::BaseDirection;

    round_trip(
        &Literal::new_directional_language_tagged_literal("foo", "en", BaseDirection::Rtl)?,
        r#"{"value":"foo","language":"en","direction":"rtl"}"#,
    )?;
    let triple = Triple::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        NamedNode::new("http://example.com/o")?,
    );
    round_trip(
        &Term::Triple(Box::new(triple)),
        concat!(
            r#"{"type":"triple","subject":{"type":"uri","value":"http://example.com/s"},"#,
            r#""predicate":{"type":"uri","value":"http://example.com/p"},"#,
            r#""object":{"type":"uri","value":"http://example.com/o"}}"#
        ),
    )?;
    Ok(())
}
