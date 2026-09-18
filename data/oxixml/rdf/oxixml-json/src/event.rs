//! The [`JsonEvent`] type: one token in a streaming JSON parse.

use std::borrow::Cow;

/// A single event produced while streaming a JSON document, or consumed while serializing one.
///
/// A well-formed sequence of [`JsonEvent`]s is exactly the pre-order traversal of a JSON value's
/// syntax tree: containers ([`StartArray`](JsonEvent::StartArray)/[`StartObject`](JsonEvent::StartObject))
/// open and later close, object members are announced with [`ObjectKey`](JsonEvent::ObjectKey)
/// immediately before the member's value, and the stream ends with a single
/// [`Eof`](JsonEvent::Eof).
///
/// [`String`](JsonEvent::String) and [`Number`](JsonEvent::Number) carry [`Cow<str>`] rather
/// than an already-decoded value:
///
/// * A [`Number`](JsonEvent::Number) is the raw JSON number literal (e.g. `"1.5e10"`); parsing it
///   into an `i64`/`u64`/`f64` (or an arbitrary-precision type) is left to the caller, since RFC
///   8259 does not mandate a numeric representation and different consumers need different ones.
/// * A [`String`](JsonEvent::String) is the *decoded* string content (escapes already resolved):
///   it borrows directly from the input when the source text needed no unescaping, and owns a
///   fresh [`String`] only when escape processing required allocating one.
///
/// Duplicate object keys are **not** rejected at this layer: RFC 8259 explicitly leaves the
/// handling of names within an object unspecified, so the event stream reports every
/// [`ObjectKey`](JsonEvent::ObjectKey)/value pair as written and leaves any "last key wins" or
/// "reject duplicates" policy to the consumer.
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub enum JsonEvent<'a> {
    /// A JSON string value (object member value or array element), with all escape sequences
    /// already decoded.
    String(Cow<'a, str>),
    /// A JSON number, as its raw (unparsed) literal text, e.g. `"-12.5e+3"`.
    Number(Cow<'a, str>),
    /// The JSON literals `true` and `false`.
    Boolean(bool),
    /// The JSON literal `null`.
    Null,
    /// The start of a JSON array: `[`.
    StartArray,
    /// The end of a JSON array: `]`.
    EndArray,
    /// The start of a JSON object: `{`.
    StartObject,
    /// The end of a JSON object: `}`.
    EndObject,
    /// An object member name, immediately preceding that member's value event.
    ObjectKey(Cow<'a, str>),
    /// The end of the document: exactly one root value has been fully read (or written) and no
    /// further content follows.
    Eof,
}

impl JsonEvent<'_> {
    /// Returns an owned copy of this event whose borrowed strings, if any, no longer depend on
    /// the lifetime `'a` of the source buffer.
    #[must_use]
    pub fn into_owned(self) -> JsonEvent<'static> {
        match self {
            JsonEvent::String(s) => JsonEvent::String(Cow::Owned(s.into_owned())),
            JsonEvent::Number(n) => JsonEvent::Number(Cow::Owned(n.into_owned())),
            JsonEvent::Boolean(b) => JsonEvent::Boolean(b),
            JsonEvent::Null => JsonEvent::Null,
            JsonEvent::StartArray => JsonEvent::StartArray,
            JsonEvent::EndArray => JsonEvent::EndArray,
            JsonEvent::StartObject => JsonEvent::StartObject,
            JsonEvent::EndObject => JsonEvent::EndObject,
            JsonEvent::ObjectKey(k) => JsonEvent::ObjectKey(Cow::Owned(k.into_owned())),
            JsonEvent::Eof => JsonEvent::Eof,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_owned_detaches_lifetime() {
        let source = String::from("{\"a\":1}");
        let borrowed = JsonEvent::String(Cow::Borrowed(&source[1..2]));
        let owned = borrowed.into_owned();
        drop(source);
        assert_eq!(owned, JsonEvent::String(Cow::Owned("\"".to_string())));
    }

    #[test]
    fn equality_and_clone() {
        let a = JsonEvent::Number(Cow::Borrowed("1"));
        let b = a.clone();
        assert_eq!(a, b);
        assert_ne!(JsonEvent::Null, JsonEvent::Eof);
    }
}
