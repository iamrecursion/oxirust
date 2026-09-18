//! A compact, cheaply-clonable string type used throughout the RDF data model.
//!
//! [`OxStr`] mirrors the public surface of `oxrdf`'s string type but is implemented in **safe
//! Rust** (no `unsafe`), using [`Arc<str>`](std::sync::Arc) for owned data and a plain string
//! slice for the borrowed case. Owned values clone in O(1) by bumping an atomic reference count,
//! while a `const`-constructible borrowed variant lets the vocabulary tables
//! ([`crate::vocab`]) hold [`NamedNode`](crate::NamedNode) constants with no allocation.

use std::borrow::{Borrow, Cow};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// Owned variant of [`OxStr`]: a compact string type for reference-counted owned data or static
/// slices.
///
/// `OxString` is conceptually a fusion of [`Arc<str>`](std::sync::Arc) and
/// [`Cow<'static, str>`](std::borrow::Cow): it can store a `&'static str` or a reference-counted
/// fixed-size string, enabling cheap clones of owned data.
///
/// ```
/// use oxixml_model::OxString;
///
/// let value = OxString::new("hello");
/// assert_eq!(value.as_str(), "hello");
/// ```
pub type OxString = OxStr<'static>;

/// A compact string type that is either borrowed or reference-counted owned data.
///
/// `OxStr` is conceptually a fusion of [`Arc<str>`](std::sync::Arc) and
/// [`Cow<'a, str>`](std::borrow::Cow). Cloning an owned value is cheap: it only increments an
/// atomic reference count. Cloning a borrowed value simply copies the slice pointer.
///
/// ```
/// use oxixml_model::OxStr;
///
/// let borrowed = OxStr::new("hello");
/// let owned = OxStr::new_owned("hello");
///
/// assert_eq!(borrowed.as_str(), "hello");
/// assert_eq!(owned.as_str(), "hello");
/// ```
#[derive(Clone)]
pub struct OxStr<'a>(Repr<'a>);

#[derive(Clone)]
enum Repr<'a> {
    Borrowed(&'a str),
    Shared(Arc<str>),
}

impl<'a> OxStr<'a> {
    /// Creates an `OxStr` borrowing from `value`.
    ///
    /// This does not allocate and keeps the lifetime of `value`.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let value = OxStr::new("abc");
    /// assert_eq!(value.as_str(), "abc");
    /// ```
    #[inline]
    pub const fn new(value: &'a str) -> Self {
        Self(Repr::Borrowed(value))
    }

    /// Creates an owned `OxStr` by copying `value`.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let value = OxStr::new_owned("abc");
    /// assert_eq!(value.as_str(), "abc");
    /// ```
    #[inline]
    pub fn new_owned(value: &str) -> Self {
        Self(Repr::Shared(Arc::from(value)))
    }

    /// The fallible counterpart of [`new_owned`](Self::new_owned).
    ///
    /// Returns `None` instead of aborting when the buffer holding the copy cannot be allocated,
    /// or when the value's length is not representable.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let value = OxStr::try_new_owned("abc").expect("small enough to allocate");
    /// assert_eq!(value.as_str(), "abc");
    /// ```
    #[inline]
    pub fn try_new_owned(value: &str) -> Option<Self> {
        Self::try_concat([value])
    }

    /// Concatenates all `values` into a new owned `OxStr`.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let value = OxStr::concat(["ab", "cd", "ef"]);
    /// assert_eq!(value.as_str(), "abcdef");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the concatenated length overflows [`usize`] or the buffer cannot be allocated;
    /// use [`try_concat`](Self::try_concat) to handle that case.
    #[inline]
    pub fn concat<T: AsRef<str>>(values: impl AsRef<[T]>) -> Self {
        Self::try_concat(values).expect("OxStr::concat: the concatenated value is not allocatable")
    }

    /// The fallible counterpart of [`concat`](Self::concat).
    ///
    /// Returns `None` if the total length is not representable as a [`usize`], or if the buffer
    /// holding the concatenation cannot be allocated.
    ///
    /// Note that the final, reference-counted copy is made with [`Arc`]'s infallible constructor,
    /// which the standard library gives no fallible counterpart for: an allocation failure at
    /// that very last step still aborts.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let value = OxStr::try_concat(["ab", "cd", "ef"]).expect("small enough to allocate");
    /// assert_eq!(value.as_str(), "abcdef");
    /// assert_eq!(OxStr::try_concat::<&str>([]).map(|v| v.as_str().to_owned()), Some(String::new()));
    /// ```
    pub fn try_concat<T: AsRef<str>>(values: impl AsRef<[T]>) -> Option<Self> {
        let values = values.as_ref();
        let capacity = values.iter().try_fold(0usize, |total, value| {
            total.checked_add(value.as_ref().len())
        })?;
        let mut buffer = String::new();
        buffer.try_reserve_exact(capacity).ok()?;
        for value in values {
            buffer.push_str(value.as_ref());
        }
        Some(Self(Repr::Shared(Arc::from(buffer.as_str()))))
    }

    /// Converts to an owned [`OxStr<'static>`](Self).
    ///
    /// If `self` is already owned, this only increments the internal atomic reference count.
    /// If `self` is borrowed, data is copied into a new allocation.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let borrowed = OxStr::new("hello");
    /// let owned = borrowed.to_owned();
    /// assert_eq!(owned.as_str(), "hello");
    /// ```
    #[inline]
    pub fn to_owned(&self) -> OxString {
        match &self.0 {
            Repr::Borrowed(value) => OxStr(Repr::Shared(Arc::from(*value))),
            Repr::Shared(value) => OxStr(Repr::Shared(Arc::clone(value))),
        }
    }

    /// Returns the inner string as a slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        match &self.0 {
            Repr::Borrowed(value) => value,
            Repr::Shared(value) => value,
        }
    }

    /// Returns a mutable view of the string contents if this value is owned and uniquely held.
    ///
    /// Returns `None` for borrowed values and for shared owned values (reference count > 1).
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let mut borrowed = OxStr::new("abc");
    /// assert_eq!(borrowed.get_mut(), None);
    ///
    /// let mut owned = OxStr::new_owned("abc");
    /// assert_eq!(owned.get_mut().as_deref(), Some("abc"));
    /// ```
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut str> {
        match &mut self.0 {
            Repr::Borrowed(_) => None,
            Repr::Shared(value) => Arc::get_mut(value),
        }
    }

    /// Ensures unique ownership and returns a mutable view of the string contents.
    ///
    /// If `self` is borrowed or shared, this first clones the string data (copy-on-write). If it
    /// is already uniquely owned, no allocation is performed.
    ///
    /// ```
    /// use oxixml_model::OxStr;
    ///
    /// let value = OxStr::new("abc");
    /// let mut copy = value.clone();
    /// copy.make_mut().make_ascii_uppercase();
    ///
    /// assert_eq!(value.as_str(), "abc");
    /// assert_eq!(copy.as_str(), "ABC");
    /// ```
    #[inline]
    pub fn make_mut(&mut self) -> &mut str {
        let needs_clone = match &mut self.0 {
            Repr::Borrowed(_) => true,
            Repr::Shared(value) => Arc::get_mut(value).is_none(),
        };
        if needs_clone {
            self.0 = Repr::Shared(Arc::from(self.as_str()));
        }
        match &mut self.0 {
            Repr::Shared(value) => {
                Arc::get_mut(value).expect("make_mut ensured a uniquely-owned Arc above")
            }
            Repr::Borrowed(_) => {
                unreachable!("make_mut converted any borrowed value into an owned Arc above")
            }
        }
    }
}

// `Send`/`Sync` are provided automatically by the `Arc<str>` and `&str` fields.

impl Default for OxStr<'_> {
    #[inline]
    fn default() -> Self {
        Self(Repr::Borrowed(""))
    }
}

impl AsRef<str> for OxStr<'_> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Deref for OxStr<'_> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for OxStr<'_> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for OxStr<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str().eq(other.as_str())
    }
}

impl PartialEq<str> for OxStr<'_> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str().eq(other)
    }
}

impl PartialEq<&str> for OxStr<'_> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str().eq(*other)
    }
}

impl PartialEq<String> for OxStr<'_> {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str().eq(other.as_str())
    }
}

impl PartialEq<Cow<'_, str>> for OxStr<'_> {
    #[inline]
    fn eq(&self, other: &Cow<'_, str>) -> bool {
        self.as_str().eq(other.as_ref())
    }
}

impl PartialEq<OxStr<'_>> for str {
    #[inline]
    fn eq(&self, other: &OxStr<'_>) -> bool {
        self.eq(other.as_str())
    }
}

impl PartialEq<OxStr<'_>> for &str {
    #[inline]
    fn eq(&self, other: &OxStr<'_>) -> bool {
        (*self).eq(other.as_str())
    }
}

impl PartialEq<OxStr<'_>> for String {
    #[inline]
    fn eq(&self, other: &OxStr<'_>) -> bool {
        self.as_str().eq(other.as_str())
    }
}

impl PartialEq<OxStr<'_>> for Cow<'_, str> {
    #[inline]
    fn eq(&self, other: &OxStr<'_>) -> bool {
        self.as_ref().eq(other.as_str())
    }
}

impl Eq for OxStr<'_> {}

impl PartialOrd for OxStr<'_> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OxStr<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for OxStr<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl fmt::Debug for OxStr<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for OxStr<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

/// Conversion into an owned [`OxString`], accepting borrowed values of any lifetime.
///
/// [`OxString`] is an alias for [`OxStr<'static>`], so a plain `impl Into<OxString>` bound only
/// accepts string slices that live for `'static`. This trait is the bound used by the term
/// constructors ([`NamedNode::new`](crate::NamedNode::new),
/// [`Literal::new_typed_literal`](crate::Literal::new_typed_literal), …) so that they accept any
/// `&str`, exactly like the `impl Into<String>` bounds of their `oxrdf` counterparts.
///
/// Borrowed input is copied into a reference-counted buffer; input that is already owned
/// ([`OxString`], [`String`], [`Box<str>`]) is moved without a new allocation, and an [`OxString`]
/// that borrows a `'static` slice keeps borrowing it.
///
/// ```
/// use oxixml_model::{NamedNode, OxString};
///
/// // A `&str` that is not `'static` is accepted.
/// let owned = String::from("http://example.com");
/// let borrowed: &str = &owned;
/// assert_eq!(NamedNode::new(borrowed)?.as_str(), "http://example.com");
///
/// // An `OxString` is moved through unchanged.
/// assert_eq!(NamedNode::new(OxString::new("http://example.com"))?.as_str(), "http://example.com");
/// # Result::<_, oxixml_model::IriParseError>::Ok(())
/// ```
pub trait IntoOxString {
    /// Performs the conversion.
    fn into_ox_string(self) -> OxString;
}

impl IntoOxString for OxString {
    #[inline]
    fn into_ox_string(self) -> OxString {
        self
    }
}

impl IntoOxString for &str {
    #[inline]
    fn into_ox_string(self) -> OxString {
        OxString::new_owned(self)
    }
}

impl IntoOxString for &String {
    #[inline]
    fn into_ox_string(self) -> OxString {
        OxString::new_owned(self)
    }
}

impl IntoOxString for String {
    #[inline]
    fn into_ox_string(self) -> OxString {
        OxString::from(self)
    }
}

impl IntoOxString for Box<str> {
    #[inline]
    fn into_ox_string(self) -> OxString {
        OxString::from(String::from(self))
    }
}

impl IntoOxString for Cow<'_, str> {
    #[inline]
    fn into_ox_string(self) -> OxString {
        match self {
            Cow::Borrowed(value) => OxString::new_owned(value),
            Cow::Owned(value) => OxString::from(value),
        }
    }
}

impl<'a> From<&'a str> for OxStr<'a> {
    #[inline]
    fn from(value: &'a str) -> Self {
        Self::new(value)
    }
}

impl From<String> for OxStr<'_> {
    #[inline]
    fn from(value: String) -> Self {
        Self(Repr::Shared(Arc::from(value)))
    }
}

impl<'a> From<Cow<'a, str>> for OxStr<'a> {
    #[inline]
    fn from(value: Cow<'a, str>) -> Self {
        match value {
            Cow::Borrowed(value) => Self::new(value),
            Cow::Owned(value) => Self::from(value),
        }
    }
}

impl From<OxStr<'_>> for String {
    #[inline]
    fn from(value: OxStr<'_>) -> Self {
        value.as_str().to_owned()
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::{OxStr, fmt};
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

    impl Serialize for OxStr<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.as_str().serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for OxStr<'_> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct StrVisitor;

            impl de::Visitor<'_> for StrVisitor {
                type Value = OxStr<'static>;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a string")
                }

                fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                    Ok(OxStr::new_owned(value))
                }

                fn visit_bytes<E: de::Error>(self, value: &[u8]) -> Result<Self::Value, E> {
                    let value = std::str::from_utf8(value).map_err(|_| {
                        de::Error::invalid_value(de::Unexpected::Bytes(value), &self)
                    })?;
                    self.visit_str(value)
                }
            }

            deserializer.deserialize_str(StrVisitor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_clone() {
        let value = OxStr::new_owned("a");
        let copy = value.clone();
        drop(value);
        assert_eq!(copy.as_str(), "a");
    }

    #[test]
    fn as_mut() {
        assert_eq!(OxStr::new("a").get_mut(), None);
        let mut owned = OxStr::new_owned("a");
        assert_eq!(owned.get_mut().as_deref(), Some("a"));
        assert_eq!(owned.clone().get_mut(), None);
    }

    #[test]
    fn make_mut() {
        let slice = OxStr::new("a");
        let mut slice_clone = slice.clone();
        slice_clone.make_mut().make_ascii_uppercase();
        assert_eq!(slice.as_str(), "a");
        assert_eq!(slice_clone.as_str(), "A");

        let mut owned = OxStr::new_owned("a");
        owned.make_mut().make_ascii_uppercase();
        assert_eq!(owned.as_str(), "A");

        let mut owned_clone = owned.clone();
        owned_clone.make_mut().make_ascii_lowercase();
        assert_eq!(owned.as_str(), "A");
        assert_eq!(owned_clone.as_str(), "a");
    }

    #[test]
    fn default_is_empty() {
        assert_eq!(OxStr::default(), "");
    }

    #[test]
    fn concat() {
        assert_eq!(OxStr::concat(["ab", "cd", "ef"]).as_str(), "abcdef");
    }

    /// The fallible constructors agree with their panicking counterparts, and reject a total
    /// length that is not representable.
    #[test]
    fn try_constructors() {
        assert_eq!(
            OxStr::try_new_owned("abc").map(|v| v.as_str().to_owned()),
            Some("abc".to_owned())
        );
        assert_eq!(
            OxStr::try_concat(["ab", "cd", "ef"]).map(|v| v.as_str().to_owned()),
            Some("abcdef".to_owned())
        );
        assert_eq!(
            OxStr::try_concat::<&str>([]).map(|v| v.as_str().to_owned()),
            Some(String::new())
        );

        // `concat` is now a thin wrapper over `try_concat` and must behave identically.
        assert_eq!(OxStr::concat(["ab", "cd"]).as_str(), "abcd");
        assert_eq!(OxStr::new_owned("abc").as_str(), "abc");
    }

    /// Every input shape accepted by `oxrdf`'s `impl Into<String>` constructor bounds is accepted
    /// here too, in particular a `&str` that does not live for `'static`.
    #[test]
    fn into_ox_string_accepts_every_string_shape() {
        let owned = String::from("abc");
        let borrowed: &str = &owned;

        assert_eq!(borrowed.into_ox_string(), "abc");
        assert_eq!((&owned).into_ox_string(), "abc");
        assert_eq!(owned.clone().into_ox_string(), "abc");
        assert_eq!(Box::<str>::from("abc").into_ox_string(), "abc");
        assert_eq!(Cow::Borrowed(borrowed).into_ox_string(), "abc");
        assert_eq!(Cow::<str>::Owned(owned).into_ox_string(), "abc");
        assert_eq!(OxString::new("abc").into_ox_string(), "abc");
    }

    /// An already-owned [`OxString`] is moved through untouched, so a `'static` slice keeps being
    /// borrowed instead of being copied into a fresh allocation.
    #[test]
    fn into_ox_string_keeps_zero_copy_inputs_borrowed() {
        assert!(matches!(
            OxString::new("abc").into_ox_string().0,
            Repr::Borrowed(_)
        ));
        assert!(matches!(
            String::from("abc").into_ox_string().0,
            Repr::Shared(_)
        ));
    }

    /// The term constructors accept a non-`'static` `&str`, exactly like their `oxrdf`
    /// counterparts. This is a compile-time property first and a value check second.
    #[test]
    fn term_constructors_accept_non_static_str() {
        let owned = String::from("http://example.com/a");
        let iri: &str = &owned;
        let node = crate::NamedNode::new(iri).expect("the IRI is valid");
        assert_eq!(node.as_str(), "http://example.com/a");
        assert_eq!(crate::NamedNode::new_unchecked(iri).as_str(), iri);

        let text = String::from("hello");
        let value: &str = &text;
        assert_eq!(crate::Literal::new_simple_literal(value).value(), "hello");
        assert_eq!(
            crate::Literal::new_typed_literal(value, node).value(),
            "hello"
        );
        assert_eq!(
            crate::Literal::new_language_tagged_literal(value, "en")
                .expect("the language tag is valid")
                .value(),
            "hello"
        );

        let id = String::from("b0");
        let id: &str = &id;
        assert_eq!(
            crate::BlankNode::new(id).expect("the id is valid").as_str(),
            "b0"
        );

        let name = String::from("v");
        let name: &str = &name;
        assert_eq!(
            crate::Variable::new(name)
                .expect("the name is valid")
                .as_str(),
            "v"
        );
    }
}
