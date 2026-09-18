//! The internal JSON value tree used by every JSON-LD algorithm in this crate.
//!
//! JSON-LD is defined over an abstract JSON data model rather than over a byte stream, so this
//! crate materialises each document into a [`JsonValue`] tree (built from
//! [`oxixml_json`](oxixml_json) events) and runs the specification algorithms — context
//! processing, expansion, compaction, flattening, and RDF (de)serialization — over that tree.
//!
//! Object member order is **preserved**: [`JsonObject`] keeps insertion order so that
//! serialization is deterministic and so that the few algorithms whose output depends on member
//! order (blank-node identifier allocation, the streaming profile's key-order check) behave
//! reproducibly. Duplicate keys keep the position of the first occurrence but take the value of
//! the last, matching the "last value wins" convention of mainstream JSON parsers.

use oxixml_json::{
    JsonEvent, JsonParseError, JsonSyntaxError, ReaderJsonParser, SliceJsonParser,
    WriterJsonSerializer,
};
use std::borrow::Cow;
use std::io::{Read, Write};

/// The maximum number of nested arrays/objects [`JsonValue::from_slice`] and
/// [`JsonValue::from_reader`] accept by default.
///
/// Every JSON-LD algorithm in this crate (expansion, compaction, flattening, canonicalization,
/// serialization) walks a [`JsonValue`] tree with one native stack frame per nesting level, so an
/// unbounded tree turns an untrusted, modestly-sized document into a process-aborting stack
/// overflow. The bound is deliberately far below the underlying JSON tokenizer's own
/// container-depth cap, which is calibrated for a purely iterative consumer. `oxixml-turtle`
/// faces the same risk class from nested collections and blank-node property lists but sidesteps
/// it differently: its parsers hold parser state in a heap-allocated stack (`Vec<N3State>` /
/// `Vec<TriGState>`) rather than in native call frames, so it has no comparable depth to cap and
/// accepts arbitrarily deep nesting, bounded only by available memory.
pub const DEFAULT_MAX_NESTING_DEPTH: usize = 128;

/// An in-memory JSON value: the abstract data model every JSON-LD algorithm operates on.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    /// The JSON literal `null`.
    Null,
    /// A JSON boolean (`true` / `false`).
    Bool(bool),
    /// A JSON number, kept as its raw literal text so no precision is lost before the algorithm
    /// that consumes it decides how to interpret it.
    Number(String),
    /// A JSON string, with all escapes already decoded.
    String(String),
    /// A JSON array.
    Array(Vec<JsonValue>),
    /// A JSON object, with member order preserved (see the module documentation).
    Object(JsonObject),
}

/// An insertion-ordered JSON object (map of string keys to [`JsonValue`]s).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JsonObject {
    entries: Vec<(String, JsonValue)>,
}

impl JsonObject {
    /// Creates an empty object.
    #[must_use]
    pub fn new() -> Self {
        Self::empty()
    }

    /// Creates an empty object in a `const` context.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the value associated with `key`, if any.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Returns a mutable reference to the value associated with `key`, if any.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut JsonValue> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Whether the object contains `key`.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Inserts `key`/`value`, replacing (in place) any existing value for `key`.
    ///
    /// Returns the previous value, if the key was already present.
    pub fn insert(&mut self, key: impl Into<String>, value: JsonValue) -> Option<JsonValue> {
        let key = key.into();
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(std::mem::replace(&mut slot.1, value))
        } else {
            self.entries.push((key, value));
            None
        }
    }

    /// Removes `key`, returning its value if present.
    pub fn remove(&mut self, key: &str) -> Option<JsonValue> {
        let idx = self.entries.iter().position(|(k, _)| k == key)?;
        Some(self.entries.remove(idx).1)
    }

    /// Iterates over the members in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &JsonValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterates mutably over the members in insertion order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut JsonValue)> {
        self.entries.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterates over the member keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    /// The member keys, sorted lexicographically by Unicode code point.
    #[must_use]
    pub fn sorted_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.entries.iter().map(|(k, _)| k.clone()).collect();
        keys.sort_unstable();
        keys
    }

    /// Consumes the object, yielding its members in insertion order.
    #[must_use]
    pub fn into_entries(self) -> Vec<(String, JsonValue)> {
        self.entries
    }
}

impl FromIterator<(String, JsonValue)> for JsonObject {
    fn from_iter<T: IntoIterator<Item = (String, JsonValue)>>(iter: T) -> Self {
        let mut object = JsonObject::new();
        for (k, v) in iter {
            object.insert(k, v);
        }
        object
    }
}

impl JsonValue {
    /// Whether this value is an array.
    #[must_use]
    pub fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array(_))
    }

    /// Whether this value is an object.
    #[must_use]
    pub fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Whether this value is a JSON string.
    #[must_use]
    pub fn is_string(&self) -> bool {
        matches!(self, JsonValue::String(_))
    }

    /// Whether this value is a scalar (string, number, or boolean — i.e. not null, array or
    /// object).
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            JsonValue::String(_) | JsonValue::Number(_) | JsonValue::Bool(_)
        )
    }

    /// Borrows this value as a string, if it is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrows this value as an object, if it is one.
    #[must_use]
    pub fn as_object(&self) -> Option<&JsonObject> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Mutably borrows this value as an object, if it is one.
    pub fn as_object_mut(&mut self) -> Option<&mut JsonObject> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Borrows this value as an array, if it is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Parses a JSON document already fully in memory into a [`JsonValue`] tree, rejecting
    /// documents nested more than [`DEFAULT_MAX_NESTING_DEPTH`] levels deep.
    pub fn from_slice(data: &[u8]) -> Result<Self, JsonSyntaxError> {
        Self::from_slice_with_max_nesting_depth(data, DEFAULT_MAX_NESTING_DEPTH)
    }

    /// Same as [`from_slice`](Self::from_slice) with an explicit nesting-depth bound.
    ///
    /// Raising the bound is only safe for a consumer that never walks the resulting tree
    /// recursively; see [`DEFAULT_MAX_NESTING_DEPTH`].
    pub fn from_slice_with_max_nesting_depth(
        data: &[u8],
        max_nesting_depth: usize,
    ) -> Result<Self, JsonSyntaxError> {
        let mut parser = SliceJsonParser::new(data).with_max_stack_size(max_nesting_depth);
        let mut builder = TreeBuilder::default();
        loop {
            let event = parser.parse_next()?;
            if matches!(event, JsonEvent::Eof) {
                break;
            }
            builder.push(event)?;
        }
        builder.finish()
    }

    /// Parses a JSON document from a [`Read`] source into a [`JsonValue`] tree, rejecting
    /// documents nested more than [`DEFAULT_MAX_NESTING_DEPTH`] levels deep.
    pub fn from_reader(reader: impl Read) -> Result<Self, JsonParseError> {
        Self::from_reader_with_max_nesting_depth(reader, DEFAULT_MAX_NESTING_DEPTH)
    }

    /// Same as [`from_reader`](Self::from_reader) with an explicit nesting-depth bound.
    pub fn from_reader_with_max_nesting_depth(
        reader: impl Read,
        max_nesting_depth: usize,
    ) -> Result<Self, JsonParseError> {
        let mut parser = ReaderJsonParser::new(reader).with_max_stack_size(max_nesting_depth);
        let mut builder = TreeBuilder::default();
        loop {
            let event = parser.parse_next()?;
            if matches!(event, JsonEvent::Eof) {
                break;
            }
            builder.push(event).map_err(JsonParseError::Syntax)?;
        }
        builder.finish().map_err(JsonParseError::Syntax)
    }

    /// Serializes this value to `writer` as compact JSON (no insignificant whitespace).
    pub fn write_to(&self, writer: impl Write) -> std::io::Result<()> {
        let mut serializer = WriterJsonSerializer::new(writer);
        self.emit(&mut serializer)?;
        serializer.finish()?;
        Ok(())
    }

    /// Serializes this value to an owned `String` of compact JSON.
    pub fn to_json_string(&self) -> String {
        let mut buffer = Vec::new();
        // Writing to a `Vec<u8>` is infallible.
        if self.write_to(&mut buffer).is_err() {
            return String::new();
        }
        String::from_utf8(buffer).unwrap_or_default()
    }

    /// Writes this value out **iteratively**: an explicit work stack, not native recursion, so
    /// that serializing a deeply nested tree can never overflow the stack.
    fn emit<W: Write>(&self, serializer: &mut WriterJsonSerializer<W>) -> std::io::Result<()> {
        enum Step<'a> {
            Value(&'a JsonValue),
            Key(&'a str),
            EndArray,
            EndObject,
        }

        let mut stack = vec![Step::Value(self)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Key(key) => {
                    serializer.serialize_event(JsonEvent::ObjectKey(Cow::Borrowed(key)))?;
                }
                Step::EndArray => serializer.serialize_event(JsonEvent::EndArray)?,
                Step::EndObject => serializer.serialize_event(JsonEvent::EndObject)?,
                Step::Value(JsonValue::Null) => serializer.serialize_event(JsonEvent::Null)?,
                Step::Value(JsonValue::Bool(b)) => {
                    serializer.serialize_event(JsonEvent::Boolean(*b))?;
                }
                Step::Value(JsonValue::Number(n)) => {
                    serializer.serialize_event(JsonEvent::Number(Cow::Borrowed(n)))?;
                }
                Step::Value(JsonValue::String(s)) => {
                    serializer.serialize_event(JsonEvent::String(Cow::Borrowed(s)))?;
                }
                Step::Value(JsonValue::Array(items)) => {
                    serializer.serialize_event(JsonEvent::StartArray)?;
                    stack.push(Step::EndArray);
                    stack.extend(items.iter().rev().map(Step::Value));
                }
                Step::Value(JsonValue::Object(object)) => {
                    serializer.serialize_event(JsonEvent::StartObject)?;
                    stack.push(Step::EndObject);
                    for (key, value) in object.entries.iter().rev() {
                        stack.push(Step::Value(value));
                        stack.push(Step::Key(key));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Incrementally assembles a [`JsonValue`] tree from a flat [`JsonEvent`] stream.
#[derive(Default)]
struct TreeBuilder {
    stack: Vec<Frame>,
    root: Option<JsonValue>,
}

enum Frame {
    Array(Vec<JsonValue>),
    Object(JsonObject, Option<String>),
}

impl TreeBuilder {
    fn push(&mut self, event: JsonEvent<'_>) -> Result<(), JsonSyntaxError> {
        match event {
            JsonEvent::StartArray => self.stack.push(Frame::Array(Vec::new())),
            JsonEvent::StartObject => self.stack.push(Frame::Object(JsonObject::new(), None)),
            JsonEvent::EndArray => {
                if let Some(Frame::Array(items)) = self.stack.pop() {
                    self.add_value(JsonValue::Array(items));
                }
            }
            JsonEvent::EndObject => {
                if let Some(Frame::Object(object, _)) = self.stack.pop() {
                    self.add_value(JsonValue::Object(object));
                }
            }
            JsonEvent::ObjectKey(key) => {
                if let Some(Frame::Object(_, pending)) = self.stack.last_mut() {
                    *pending = Some(key.into_owned());
                }
            }
            JsonEvent::String(s) => self.add_value(JsonValue::String(s.into_owned())),
            JsonEvent::Number(n) => self.add_value(JsonValue::Number(n.into_owned())),
            JsonEvent::Boolean(b) => self.add_value(JsonValue::Bool(b)),
            JsonEvent::Null => self.add_value(JsonValue::Null),
            JsonEvent::Eof => {}
        }
        Ok(())
    }

    fn add_value(&mut self, value: JsonValue) {
        match self.stack.last_mut() {
            Some(Frame::Array(items)) => items.push(value),
            Some(Frame::Object(object, pending)) => {
                if let Some(key) = pending.take() {
                    object.insert(key, value);
                }
            }
            None => self.root = Some(value),
        }
    }

    fn finish(self) -> Result<JsonValue, JsonSyntaxError> {
        Ok(self.root.unwrap_or(JsonValue::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_object() -> Result<(), JsonSyntaxError> {
        let value = JsonValue::from_slice(br#"{"b":1,"a":[true,null,"x"]}"#)?;
        assert_eq!(value.to_json_string(), r#"{"b":1,"a":[true,null,"x"]}"#);
        Ok(())
    }

    #[test]
    fn duplicate_keys_last_value_wins_first_position() -> Result<(), JsonSyntaxError> {
        let value = JsonValue::from_slice(br#"{"a":1,"b":2,"a":3}"#)?;
        assert_eq!(value.to_json_string(), r#"{"a":3,"b":2}"#);
        Ok(())
    }

    #[test]
    fn sorted_keys_are_code_point_ordered() {
        let mut object = JsonObject::new();
        object.insert("b", JsonValue::Null);
        object.insert("a", JsonValue::Null);
        object.insert("Z", JsonValue::Null);
        assert_eq!(object.sorted_keys(), vec!["Z", "a", "b"]);
    }

    /// Builds `{"p":{"p":{...:"leaf"}}}` nested `depth` levels deep.
    fn nested_document(depth: usize) -> Vec<u8> {
        let mut document = Vec::new();
        for _ in 0..depth {
            document.extend_from_slice(br#"{"p":"#);
        }
        document.extend_from_slice(br#""leaf""#);
        document.extend(std::iter::repeat_n(b'}', depth));
        document
    }

    /// Regression test: a deeply nested document must be rejected with an error rather than
    /// aborting the process with a stack overflow once something walks the resulting tree.
    #[test]
    fn rejects_documents_nested_past_the_limit() {
        for depth in [DEFAULT_MAX_NESTING_DEPTH * 2, 2_000, 60_000, 200_000] {
            assert!(
                JsonValue::from_slice(&nested_document(depth)).is_err(),
                "depth {depth} should be rejected"
            );
            assert!(
                JsonValue::from_reader(nested_document(depth).as_slice()).is_err(),
                "depth {depth} should be rejected"
            );
        }
    }

    /// Documents at the limit still parse, and round-trip through the (iterative) serializer.
    #[test]
    fn accepts_documents_at_the_limit() -> Result<(), JsonSyntaxError> {
        let document = nested_document(DEFAULT_MAX_NESTING_DEPTH);
        let value = JsonValue::from_slice(&document)?;
        assert_eq!(value.to_json_string().as_bytes(), document.as_slice());
        Ok(())
    }

    /// The serializer walks an explicit stack, so it survives a tree far deeper than the native
    /// stack could handle recursively (such a tree can only be built programmatically).
    #[test]
    fn serialization_of_a_very_deep_tree_does_not_overflow() {
        const DEPTH: usize = 50_000;
        let mut value = JsonValue::String("leaf".into());
        for _ in 0..DEPTH {
            value = JsonValue::Array(vec![value]);
        }
        let json = value.to_json_string();
        // Take the tree apart iteratively: `Drop` is still the compiler's recursive glue, so a
        // tree this deep must never be dropped in one go (not even while unwinding an assertion).
        let mut current = value;
        while let JsonValue::Array(mut items) = current {
            let Some(inner) = items.pop() else { break };
            current = inner;
        }
        assert_eq!(json.len(), 2 * DEPTH + 6);
        assert_eq!(&json[..DEPTH], "[".repeat(DEPTH));
        assert_eq!(&json[DEPTH..DEPTH + 6], r#""leaf""#);
        assert_eq!(&json[DEPTH + 6..], "]".repeat(DEPTH));
    }

    #[test]
    fn scalar_root() -> Result<(), JsonSyntaxError> {
        assert_eq!(
            JsonValue::from_slice(b"42")?,
            JsonValue::Number("42".into())
        );
        Ok(())
    }
}
