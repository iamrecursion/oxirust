//! RDF [blank nodes](https://www.w3.org/TR/rdf11-concepts/#dfn-blank-node): [`BlankNode`] and
//! [`BlankNodeRef`].

use crate::string::{IntoOxString, OxString};
use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fmt, str};

use oxixml_support::rng::{Rng, SplitMix64};

/// An owned RDF [blank node](https://www.w3.org/TR/rdf11-concepts/#dfn-blank-node).
///
/// The common way to create a fresh blank node is [`BlankNode::default`], which produces a
/// random, collision-free identifier.
///
/// A blank node can also be created from an explicit identifier with [`BlankNode::new`]. The
/// identifier must be valid according to the N-Triples, Turtle, and SPARQL grammars.
///
/// The default string formatter returns an N-Triples, Turtle, and SPARQL compatible
/// representation:
/// ```
/// use oxixml_model::BlankNode;
///
/// assert_eq!("_:a122", BlankNode::new("a122")?.to_string());
/// # Result::<_, oxixml_model::BlankNodeIdParseError>::Ok(())
/// ```
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct BlankNode(BlankNodeContent);

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
enum BlankNodeContent {
    Named(OxString),
    Anonymous { id: u128, str: IdStr },
}

impl BlankNode {
    /// Creates a blank node from a unique identifier.
    ///
    /// The blank node identifier must be valid according to the N-Triples, Turtle, and SPARQL
    /// grammars.
    ///
    /// In most cases it is more convenient to create a blank node with [`BlankNode::default`],
    /// which produces a random ID that can be easily inlined by stores.
    pub fn new(id: impl IntoOxString) -> Result<Self, BlankNodeIdParseError> {
        let id = id.into_ox_string();
        validate_blank_node_identifier(&id)?;
        Ok(Self::new_unchecked(id))
    }

    /// Creates a blank node from a unique identifier without validation.
    ///
    /// It is the caller's responsibility to ensure that `id` is a valid blank node identifier
    /// according to the N-Triples, Turtle, and SPARQL grammars.
    ///
    /// [`BlankNode::new`] is a safe version of this constructor and should be used for untrusted
    /// data.
    #[inline]
    pub fn new_unchecked(id: impl IntoOxString) -> Self {
        let id = id.into_ox_string();
        if let Some(numerical_id) = to_integer_id(&id) {
            Self::new_from_unique_id(numerical_id)
        } else {
            Self(BlankNodeContent::Named(id))
        }
    }

    /// Creates a blank node from a unique numerical id.
    ///
    /// In most cases it is more convenient to create a blank node with [`BlankNode::default`].
    #[inline]
    pub fn new_from_unique_id(id: u128) -> Self {
        Self(BlankNodeContent::Anonymous {
            id,
            str: IdStr::new(id),
        })
    }

    /// Returns the underlying ID of this blank node.
    #[inline]
    pub fn as_str(&self) -> &str {
        match &self.0 {
            BlankNodeContent::Named(id) => id.as_str(),
            BlankNodeContent::Anonymous { str, .. } => str.as_str(),
        }
    }

    /// Consumes this blank node and returns its underlying ID.
    #[inline]
    pub fn into_string(self) -> OxString {
        match self.0 {
            BlankNodeContent::Named(id) => id,
            BlankNodeContent::Anonymous { str, .. } => OxString::new_owned(str.as_str()),
        }
    }

    /// Borrows this blank node as a [`BlankNodeRef`].
    #[inline]
    pub fn as_ref(&self) -> BlankNodeRef<'_> {
        BlankNodeRef(match &self.0 {
            BlankNodeContent::Named(id) => BlankNodeRefContent::Named(id.as_str()),
            BlankNodeContent::Anonymous { id, str } => BlankNodeRefContent::Anonymous {
                id: *id,
                str: str.as_str(),
            },
        })
    }

    /// Returns the internal numerical ID of this blank node if it was created using
    /// [`BlankNode::new_from_unique_id`].
    ///
    /// ```
    /// use oxixml_model::BlankNode;
    ///
    /// assert_eq!(BlankNode::new_from_unique_id(128).unique_id(), Some(128));
    /// assert_eq!(BlankNode::new("foo")?.unique_id(), None);
    /// # Result::<_, oxixml_model::BlankNodeIdParseError>::Ok(())
    /// ```
    #[inline]
    pub const fn unique_id(&self) -> Option<u128> {
        match self.0 {
            BlankNodeContent::Named(_) => None,
            BlankNodeContent::Anonymous { id, .. } => Some(id),
        }
    }
}

impl fmt::Display for BlankNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl Default for BlankNode {
    /// Builds a new RDF [blank node](https://www.w3.org/TR/rdf11-concepts/#dfn-blank-node) with a
    /// unique id.
    #[inline]
    fn default() -> Self {
        // We ensure the ID does not start with a number so that it is also valid in RDF/XML.
        loop {
            let id = random_u128();
            let str = IdStr::new(id);
            if matches!(str.as_str().as_bytes().first(), Some(b'a'..=b'f')) {
                return Self(BlankNodeContent::Anonymous { id, str });
            }
        }
    }
}

/// A borrowed RDF [blank node](https://www.w3.org/TR/rdf11-concepts/#dfn-blank-node).
///
/// The common way to create a new blank node is [`BlankNode::default`].
///
/// A blank node can also be created from an explicit identifier with [`BlankNodeRef::new`]. The
/// identifier must be valid according to the N-Triples, Turtle, and SPARQL grammars.
///
/// The default string formatter returns an N-Triples, Turtle, and SPARQL compatible
/// representation:
/// ```
/// use oxixml_model::BlankNodeRef;
///
/// assert_eq!("_:a122", BlankNodeRef::new("a122")?.to_string());
/// # Result::<_, oxixml_model::BlankNodeIdParseError>::Ok(())
/// ```
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
pub struct BlankNodeRef<'a>(BlankNodeRefContent<'a>);

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
enum BlankNodeRefContent<'a> {
    Named(&'a str),
    Anonymous { id: u128, str: &'a str },
}

impl<'a> BlankNodeRef<'a> {
    /// Creates a blank node from a unique identifier.
    ///
    /// The blank node identifier must be valid according to the N-Triples, Turtle, and SPARQL
    /// grammars.
    pub fn new(id: &'a str) -> Result<Self, BlankNodeIdParseError> {
        validate_blank_node_identifier(id)?;
        Ok(Self::new_unchecked(id))
    }

    /// Creates a blank node from a unique identifier without validation.
    ///
    /// It is the caller's responsibility to ensure that `id` is a valid blank node identifier
    /// according to the N-Triples, Turtle, and SPARQL grammars.
    ///
    /// [`BlankNodeRef::new`] is a safe version of this constructor and should be used for
    /// untrusted data.
    #[inline]
    pub fn new_unchecked(id: &'a str) -> Self {
        if let Some(numerical_id) = to_integer_id(id) {
            Self(BlankNodeRefContent::Anonymous {
                id: numerical_id,
                str: id,
            })
        } else {
            Self(BlankNodeRefContent::Named(id))
        }
    }

    /// Returns the underlying ID of this blank node.
    #[inline]
    pub const fn as_str(self) -> &'a str {
        match self.0 {
            BlankNodeRefContent::Named(id) => id,
            BlankNodeRefContent::Anonymous { str, .. } => str,
        }
    }

    /// Returns the internal numerical ID of this blank node if it was created using
    /// [`BlankNode::new_from_unique_id`].
    ///
    /// ```
    /// use oxixml_model::BlankNode;
    ///
    /// assert_eq!(
    ///     BlankNode::new_from_unique_id(128).as_ref().unique_id(),
    ///     Some(128)
    /// );
    /// assert_eq!(BlankNode::new("foo")?.as_ref().unique_id(), None);
    /// # Result::<_, oxixml_model::BlankNodeIdParseError>::Ok(())
    /// ```
    #[inline]
    pub const fn unique_id(&self) -> Option<u128> {
        match self.0 {
            BlankNodeRefContent::Named(_) => None,
            BlankNodeRefContent::Anonymous { id, .. } => Some(id),
        }
    }

    /// Copies this borrowed blank node into an owned [`BlankNode`].
    #[inline]
    pub fn into_owned(self) -> BlankNode {
        BlankNode(match self.0 {
            BlankNodeRefContent::Named(id) => BlankNodeContent::Named(OxString::new_owned(id)),
            BlankNodeRefContent::Anonymous { id, .. } => BlankNodeContent::Anonymous {
                id,
                str: IdStr::new(id),
            },
        })
    }
}

impl fmt::Display for BlankNodeRef<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_:{}", self.as_str())
    }
}

impl<'a> From<&'a BlankNode> for BlankNodeRef<'a> {
    #[inline]
    fn from(node: &'a BlankNode) -> Self {
        node.as_ref()
    }
}

impl<'a> From<BlankNodeRef<'a>> for BlankNode {
    #[inline]
    fn from(node: BlankNodeRef<'a>) -> Self {
        node.into_owned()
    }
}

impl PartialEq<BlankNode> for BlankNodeRef<'_> {
    #[inline]
    fn eq(&self, other: &BlankNode) -> bool {
        *self == other.as_ref()
    }
}

impl PartialEq<BlankNodeRef<'_>> for BlankNode {
    #[inline]
    fn eq(&self, other: &BlankNodeRef<'_>) -> bool {
        self.as_ref() == *other
    }
}

/// An inline, allocation-free hexadecimal rendering of an anonymous blank node's `u128` id.
///
/// A `u128` needs at most 32 lowercase hexadecimal digits, so the value is rendered into a
/// fixed 32-byte buffer padded with trailing NUL bytes.
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
struct IdStr([u8; 32]);

impl IdStr {
    #[inline]
    fn new(id: u128) -> Self {
        let mut buffer = [0_u8; 32];
        if id == 0 {
            buffer[0] = b'0';
            return Self(buffer);
        }
        // Render least-significant digit first, then reverse into the buffer.
        let mut digits = [0_u8; 32];
        let mut remaining = id;
        let mut len = 0;
        while remaining > 0 {
            let nibble = (remaining & 0xF) as u8;
            digits[len] = if nibble < 10 {
                b'0' + nibble
            } else {
                b'a' + (nibble - 10)
            };
            remaining >>= 4;
            len += 1;
        }
        for (target, source) in buffer.iter_mut().zip(digits[..len].iter().rev()) {
            *target = *source;
        }
        Self(buffer)
    }

    #[inline]
    fn as_str(&self) -> &str {
        let len = self.0.iter().position(|byte| *byte == 0).unwrap_or(32);
        // The buffer only ever contains ASCII hexadecimal digits written by `new`.
        str::from_utf8(&self.0[..len]).unwrap_or("")
    }
}

fn validate_blank_node_identifier(id: &str) -> Result<(), BlankNodeIdParseError> {
    let mut chars = id.chars();
    let front = chars.next().ok_or(BlankNodeIdParseError)?;
    match front {
        '0'..='9'
        | '_'
        | ':'
        | 'A'..='Z'
        | 'a'..='z'
        | '\u{00C0}'..='\u{00D6}'
        | '\u{00D8}'..='\u{00F6}'
        | '\u{00F8}'..='\u{02FF}'
        | '\u{0370}'..='\u{037D}'
        | '\u{037F}'..='\u{1FFF}'
        | '\u{200C}'..='\u{200D}'
        | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}'
        | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}'
        | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}' => (),
        _ => return Err(BlankNodeIdParseError),
    }
    for c in chars {
        match c {
            '.' // validated later
            | '-'
            | '0'..='9'
            | '\u{00B7}'
            | '\u{0300}'..='\u{036F}'
            | '\u{203F}'..='\u{2040}'
            | '_'
            | ':'
            | 'A'..='Z'
            | 'a'..='z'
            | '\u{00C0}'..='\u{00D6}'
            | '\u{00D8}'..='\u{00F6}'
            | '\u{00F8}'..='\u{02FF}'
            | '\u{0370}'..='\u{037D}'
            | '\u{037F}'..='\u{1FFF}'
            | '\u{200C}'..='\u{200D}'
            | '\u{2070}'..='\u{218F}'
            | '\u{2C00}'..='\u{2FEF}'
            | '\u{3001}'..='\u{D7FF}'
            | '\u{F900}'..='\u{FDCF}'
            | '\u{FDF0}'..='\u{FFFD}'
            | '\u{10000}'..='\u{EFFFF}' => (),
            _ => return Err(BlankNodeIdParseError),
        }
    }

    // A blank node identifier can not end with a dot.
    if id.ends_with('.') {
        Err(BlankNodeIdParseError)
    } else {
        Ok(())
    }
}

#[inline]
fn to_integer_id(id: &str) -> Option<u128> {
    let digits = id.as_bytes();
    let mut value: u128 = 0;
    if let None | Some(b'0') = digits.first() {
        return None; // No empty string or leading zeros.
    }
    for digit in digits {
        value = value.checked_mul(16)?.checked_add(
            match *digit {
                b'0'..=b'9' => digit - b'0',
                b'a'..=b'f' => digit - b'a' + 10,
                _ => return None,
            }
            .into(),
        )?;
    }
    Some(value)
}

/// Generates a collision-free `u128` value for a fresh anonymous [`BlankNode`].
///
/// This is not a cryptographic random number generator: it only needs to produce distinct values.
/// A process-wide monotonic counter is mixed with a wall-clock timestamp through
/// [`SplitMix64`](oxixml_support::rng::SplitMix64) so that repeated calls (even within the same
/// nanosecond) yield different outputs, and different processes are extremely unlikely to collide.
fn random_u128() -> u128 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let seed = (nanos as u64) ^ ((nanos >> 64) as u64) ^ count.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = SplitMix64::new(seed);
    let high = rng.next_u64();
    let low = rng.next_u64();
    (u128::from(high) << 64) | u128::from(low)
}

/// An error raised while validating a [`BlankNode`] identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlankNodeIdParseError;

impl fmt::Display for BlankNodeIdParseError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("The blank node identifier is invalid")
    }
}

impl Error for BlankNodeIdParseError {}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::{BlankNode, BlankNodeRef, OxString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

    impl Serialize for BlankNode {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.as_ref().serialize(serializer)
        }
    }

    impl Serialize for BlankNodeRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            #[derive(Serialize)]
            #[serde(rename = "BlankNode")]
            struct Value<'a> {
                value: &'a str,
            }
            Value {
                value: self.as_str(),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for BlankNode {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            #[serde(rename = "BlankNode")]
            struct Value {
                value: OxString,
            }
            Self::new(Value::deserialize(deserializer)?.value).map_err(de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_partial() {
        let node = BlankNode::new_from_unique_id(0x42);
        assert_eq!(node.as_str(), "42");
    }

    #[test]
    fn as_str_full() {
        let node = BlankNode::new_from_unique_id(0x7777_6666_5555_4444_3333_2222_1111_0000);
        assert_eq!(node.as_str(), "77776666555544443333222211110000");
    }

    #[test]
    fn new_validation() {
        assert!(BlankNode::new("").is_err());
        assert!(BlankNode::new("a").is_ok());
        assert!(BlankNode::new("-").is_err());
        assert!(BlankNode::new("a-").is_ok());
        assert!(BlankNode::new(".").is_err());
        assert!(BlankNode::new("a.").is_err());
        assert!(BlankNode::new("a.a").is_ok());
    }

    #[test]
    fn new_numerical() -> Result<(), BlankNodeIdParseError> {
        assert_eq!(
            BlankNode::new("100a")?,
            BlankNode::new_from_unique_id(0x100a),
        );
        assert_ne!(
            BlankNode::new("100A")?,
            BlankNode::new_from_unique_id(0x100a)
        );
        Ok(())
    }

    #[test]
    fn test_equals() -> Result<(), BlankNodeIdParseError> {
        assert_eq!(BlankNode::new("100a")?, BlankNodeRef::new("100a")?);
        assert_eq!(BlankNode::new("zzz")?, BlankNodeRef::new("zzz")?);
        Ok(())
    }

    #[test]
    fn default_is_unique_and_alpha_prefixed() {
        let a = BlankNode::default();
        let b = BlankNode::default();
        assert_ne!(a, b);
        assert!(matches!(a.as_str().as_bytes().first(), Some(b'a'..=b'f')));
        assert!(a.unique_id().is_some());
    }
}
