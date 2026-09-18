//! [`IriRef`]: a validated RFC 3987 IRI-reference, which may be relative.

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::str::FromStr;

use crate::error::IriParseError;
use crate::positions::{IriElementsPositions, find_iri_ref_positions};
use crate::resolve::{OutputBuffer, resolve};
use crate::validate::validate_iri_ref;

/// A [RFC 3987](https://www.rfc-editor.org/rfc/rfc3987) `IRI-reference`.
///
/// Unlike [`crate::Iri`], instances of this type may be relative (they may lack a scheme). The
/// type is generic over its text storage `T` (typically `&str` for a zero-copy borrow, or
/// `String`/`Box<str>` for an owned value) — anything that dereferences to [`str`].
///
/// # Examples
///
/// ```
/// use oxixml_iri::{Iri, IriRef};
///
/// // Parse and validate a relative reference.
/// let base = IriRef::parse("../bar/baz")?;
///
/// // Resolve a further-relative reference against it.
/// let resolved = base.resolve(&IriRef::parse("bat#foo")?)?;
/// assert_eq!(resolved.into_inner(), "../bar/bat#foo");
///
/// // `IriRef`s may also be absolute.
/// assert!(IriRef::parse("http://example.com/bar/baz").is_ok());
///
/// // An `Iri` converts losslessly into an `IriRef`.
/// let _: IriRef<String> = Iri::parse("http://example.com/bar".to_owned())?.into();
/// # Ok::<(), oxixml_iri::IriParseError>(())
/// ```
#[derive(Clone, Copy)]
pub struct IriRef<T> {
    pub(crate) iri: T,
    pub(crate) positions: IriElementsPositions,
}

impl<T: Deref<Target = str>> IriRef<T> {
    /// Parses and validates `iri` as an `IRI-reference` following the grammar in
    /// [RFC 3987](https://www.rfc-editor.org/rfc/rfc3987).
    ///
    /// This takes ownership of `iri` (or borrows it, for `T = &str`) without any further
    /// allocation; the returned `IriRef` simply remembers the byte offsets of each component.
    ///
    /// Use [`parse_unchecked`](Self::parse_unchecked) instead if `iri` is already known to be
    /// valid, to skip the validation pass.
    ///
    /// # Errors
    ///
    /// Returns [`IriParseError`] if `iri` does not match the `IRI-reference` grammar.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// IriRef::parse("//example.com/bar/baz")?;
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    pub fn parse(iri: T) -> Result<Self, IriParseError> {
        let iri = Self::parse_unchecked(iri);
        validate_iri_ref(&iri)?;
        Ok(iri)
    }

    /// Builds an `IriRef` from `iri` without validating it.
    ///
    /// The component accessors and [`resolve`](Self::resolve) still behave correctly on the
    /// resulting value (they are defined purely in terms of the RFC 3986 §5.3/§3 component
    /// grammar, which this computes eagerly), but no character-class or percent-encoding
    /// checking is performed — only use this when `iri` is already known-valid, since the
    /// result of calling other methods on an invalid `IriRef` is unspecified (though never
    /// undefined behavior; this crate has no `unsafe` code).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// IriRef::parse_unchecked("//example.com/bar/baz");
    /// ```
    #[must_use]
    pub fn parse_unchecked(iri: T) -> Self {
        let positions = find_iri_ref_positions(iri.as_ref());
        Self { iri, positions }
    }

    /// Resolves `reference` against `self` as the base, following the strict
    /// [RFC 3986 §5](https://www.rfc-editor.org/rfc/rfc3986#section-5) relative-resolution
    /// algorithm, and validates the result.
    ///
    /// Use [`resolve_unchecked`](Self::resolve_unchecked) instead if both `self` and
    /// `reference` are already known-valid, to skip the validation pass.
    ///
    /// # Errors
    ///
    /// Returns [`IriParseError`] if the resolved IRI would be invalid — this can only happen
    /// when merging a base with no authority and a relative path against a `..` reference that
    /// climbs past the base's hierarchy, or when the merge would produce a path starting with
    /// `//` right after the scheme (which would be reparsed as an authority).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let base = IriRef::parse("//example.com/bar/baz")?;
    /// let resolved = base.resolve(&IriRef::parse("bat#foo")?)?;
    /// assert_eq!(resolved.into_inner(), "//example.com/bar/bat#foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    pub fn resolve<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
    ) -> Result<IriRef<String>, IriParseError> {
        let mut iri = String::new();
        let (positions, error) = resolve(self, reference, &mut iri);
        if let Some(error) = error {
            return Err(error.into());
        }
        Ok(IriRef { iri, positions })
    }

    /// Variant of [`resolve`](Self::resolve) that skips validating the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let base = IriRef::parse_unchecked("//example.com/bar/baz");
    /// let resolved = base.resolve_unchecked(&IriRef::parse_unchecked("bat#foo"));
    /// assert_eq!(resolved.into_inner(), "//example.com/bar/bat#foo");
    /// ```
    #[must_use]
    pub fn resolve_unchecked<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
    ) -> IriRef<String> {
        let mut iri = String::new();
        let (positions, _) = resolve(self, reference, &mut iri);
        IriRef { iri, positions }
    }

    /// Variant of [`resolve`](Self::resolve) that writes into `output_buffer` instead of
    /// allocating a fresh [`String`], to allow reusing a buffer across many resolutions.
    ///
    /// `output_buffer` is only appended to on success; on error its trailing content (past
    /// whatever length it started with) is unspecified.
    ///
    /// # Errors
    ///
    /// Same as [`resolve`](Self::resolve).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let base = IriRef::parse("//example.com/bar/baz")?;
    /// let mut buffer = String::new();
    /// base.resolve_into(&IriRef::parse("bat#foo")?, &mut buffer)?;
    /// assert_eq!(buffer, "//example.com/bar/bat#foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    pub fn resolve_into<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
        output_buffer: &mut impl OutputBuffer,
    ) -> Result<(), IriParseError> {
        let (_, error) = resolve(self, reference, output_buffer);
        if let Some(error) = error {
            return Err(error.into());
        }
        Ok(())
    }

    /// Variant of [`resolve_into`](Self::resolve_into) that skips validating the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let base = IriRef::parse_unchecked("//example.com/bar/baz");
    /// let mut buffer = String::new();
    /// base.resolve_into_unchecked(&IriRef::parse_unchecked("bat#foo"), &mut buffer);
    /// assert_eq!(buffer, "//example.com/bar/bat#foo");
    /// ```
    pub fn resolve_into_unchecked<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
        output_buffer: &mut impl OutputBuffer,
    ) {
        resolve(self, reference, output_buffer);
    }

    /// Returns an `IriRef<&str>` borrowing this IRI's text.
    #[inline]
    #[must_use]
    pub fn as_ref(&self) -> IriRef<&str> {
        IriRef {
            iri: &self.iri,
            positions: self.positions,
        }
    }

    /// Returns the underlying textual representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let iri = IriRef::parse("//example.com/foo")?;
    /// assert_eq!(iri.as_str(), "//example.com/foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.iri
    }

    /// Consumes `self`, returning the underlying textual representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let iri = IriRef::parse("//example.com/foo")?;
    /// assert_eq!(iri.into_inner(), "//example.com/foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    pub fn into_inner(self) -> T {
        self.iri
    }

    /// Returns `true` if this reference is absolute, i.e. it has a `scheme`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// assert!(IriRef::parse("http://example.com/foo")?.is_absolute());
    /// assert!(!IriRef::parse("/foo")?.is_absolute());
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn is_absolute(&self) -> bool {
        self.positions.scheme_end != 0
    }

    /// Returns the `scheme` component, if present.
    ///
    /// The scheme's case is preserved exactly as written; use a case-insensitive comparison
    /// (e.g. [`str::eq_ignore_ascii_case`]) to test against a specific scheme, since
    /// [RFC 3986 §3.1](https://www.rfc-editor.org/rfc/rfc3986#section-3.1) treats scheme names
    /// as case-insensitive.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let iri = IriRef::parse("hTTp://example.com")?;
    /// assert_eq!(iri.scheme(), Some("hTTp"));
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn scheme(&self) -> Option<&str> {
        if self.positions.scheme_end == 0 {
            None
        } else {
            Some(&self.iri[..self.positions.scheme_end - 1])
        }
    }

    /// Returns the `authority` component, if present.
    ///
    /// The host's case is preserved exactly as written; use a case-insensitive comparison if
    /// you look for a specific host, since [RFC 3986 §3.2.2] treats `reg-name` as
    /// case-insensitive.
    ///
    /// [RFC 3986 §3.2.2]: https://www.rfc-editor.org/rfc/rfc3986#section-3.2.2
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let http = IriRef::parse("http://user:pass@example.com:80/my/path")?;
    /// assert_eq!(http.authority(), Some("user:pass@example.com:80"));
    ///
    /// let mailto = IriRef::parse("mailto:foo@bar.com")?;
    /// assert_eq!(mailto.authority(), None);
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn authority(&self) -> Option<&str> {
        if self.positions.scheme_end + 2 > self.positions.authority_end {
            None
        } else {
            Some(&self.iri[self.positions.scheme_end + 2..self.positions.authority_end])
        }
    }

    /// Returns the `path` component. Always present, though it may be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let http = IriRef::parse("http://user:pass@example.com:80/my/path?foo=bar")?;
    /// assert_eq!(http.path(), "/my/path");
    ///
    /// let mailto = IriRef::parse("mailto:foo@bar.com")?;
    /// assert_eq!(mailto.path(), "foo@bar.com");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn path(&self) -> &str {
        &self.iri[self.positions.authority_end..self.positions.path_end]
    }

    /// Returns the `query` component, if present (excluding the leading `?`).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let iri = IriRef::parse("http://example.com/my/path?query=foo#frag")?;
    /// assert_eq!(iri.query(), Some("query=foo"));
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn query(&self) -> Option<&str> {
        if self.positions.path_end >= self.positions.query_end {
            None
        } else {
            Some(&self.iri[self.positions.path_end + 1..self.positions.query_end])
        }
    }

    /// Returns the `fragment` component, if present (excluding the leading `#`).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::IriRef;
    ///
    /// let iri = IriRef::parse("http://example.com/my/path?query=foo#frag")?;
    /// assert_eq!(iri.fragment(), Some("frag"));
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn fragment(&self) -> Option<&str> {
        if self.positions.query_end >= self.iri.len() {
            None
        } else {
            Some(&self.iri[self.positions.query_end + 1..])
        }
    }
}

impl<Lft: PartialEq<Rhs>, Rhs> PartialEq<IriRef<Rhs>> for IriRef<Lft> {
    #[inline]
    fn eq(&self, other: &IriRef<Rhs>) -> bool {
        self.iri.eq(&other.iri)
    }
}

impl<T: PartialEq<str>> PartialEq<str> for IriRef<T> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.iri.eq(other)
    }
}

impl<'a, T: PartialEq<&'a str>> PartialEq<&'a str> for IriRef<T> {
    #[inline]
    fn eq(&self, other: &&'a str) -> bool {
        self.iri.eq(other)
    }
}

impl<T: PartialEq<String>> PartialEq<String> for IriRef<T> {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.iri.eq(other)
    }
}

impl<'a, T: PartialEq<Cow<'a, str>>> PartialEq<Cow<'a, str>> for IriRef<T> {
    #[inline]
    fn eq(&self, other: &Cow<'a, str>) -> bool {
        self.iri.eq(other)
    }
}

impl<T: PartialEq<str>> PartialEq<IriRef<T>> for str {
    #[inline]
    fn eq(&self, other: &IriRef<T>) -> bool {
        other.iri.eq(self)
    }
}

impl<'a, T: PartialEq<&'a str>> PartialEq<IriRef<T>> for &'a str {
    #[inline]
    fn eq(&self, other: &IriRef<T>) -> bool {
        other.iri.eq(self)
    }
}

impl<T: PartialEq<String>> PartialEq<IriRef<T>> for String {
    #[inline]
    fn eq(&self, other: &IriRef<T>) -> bool {
        other.iri.eq(self)
    }
}

impl<'a, T: PartialEq<Cow<'a, str>>> PartialEq<IriRef<T>> for Cow<'a, str> {
    #[inline]
    fn eq(&self, other: &IriRef<T>) -> bool {
        other.iri.eq(self)
    }
}

impl<T: Eq> Eq for IriRef<T> {}

impl<T: Hash> Hash for IriRef<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.iri.hash(state);
    }
}

impl<Lft: PartialOrd<Rhs>, Rhs> PartialOrd<IriRef<Rhs>> for IriRef<Lft> {
    #[inline]
    fn partial_cmp(&self, other: &IriRef<Rhs>) -> Option<Ordering> {
        self.iri.partial_cmp(&other.iri)
    }
}

impl<T: Ord> Ord for IriRef<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iri.cmp(&other.iri)
    }
}

impl<T: Deref<Target = str>> Deref for IriRef<T> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.iri.deref()
    }
}

impl<T: AsRef<str>> AsRef<str> for IriRef<T> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.iri.as_ref()
    }
}

impl<T: Borrow<str>> Borrow<str> for IriRef<T> {
    #[inline]
    fn borrow(&self) -> &str {
        self.iri.borrow()
    }
}

impl<T: fmt::Debug> fmt::Debug for IriRef<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iri.fmt(f)
    }
}

impl<T: fmt::Display> fmt::Display for IriRef<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iri.fmt(f)
    }
}

impl FromStr for IriRef<String> {
    type Err = IriParseError;

    #[inline]
    fn from_str(iri: &str) -> Result<Self, IriParseError> {
        Self::parse(iri.to_owned())
    }
}

impl<'a> From<IriRef<&'a str>> for IriRef<String> {
    #[inline]
    fn from(iri: IriRef<&'a str>) -> Self {
        Self {
            iri: iri.iri.into(),
            positions: iri.positions,
        }
    }
}

impl<'a> From<IriRef<Cow<'a, str>>> for IriRef<String> {
    #[inline]
    fn from(iri: IriRef<Cow<'a, str>>) -> Self {
        Self {
            iri: iri.iri.into(),
            positions: iri.positions,
        }
    }
}

impl From<IriRef<Box<str>>> for IriRef<String> {
    #[inline]
    fn from(iri: IriRef<Box<str>>) -> Self {
        Self {
            iri: iri.iri.into(),
            positions: iri.positions,
        }
    }
}

impl<'a> From<IriRef<&'a str>> for IriRef<Cow<'a, str>> {
    #[inline]
    fn from(iri: IriRef<&'a str>) -> Self {
        Self {
            iri: iri.iri.into(),
            positions: iri.positions,
        }
    }
}

impl From<IriRef<String>> for IriRef<Cow<'_, str>> {
    #[inline]
    fn from(iri: IriRef<String>) -> Self {
        Self {
            iri: iri.iri.into(),
            positions: iri.positions,
        }
    }
}

impl<'a> From<&'a IriRef<String>> for IriRef<&'a str> {
    #[inline]
    fn from(iri: &'a IriRef<String>) -> Self {
        Self {
            iri: &iri.iri,
            positions: iri.positions,
        }
    }
}

impl<'a> From<&'a IriRef<Cow<'a, str>>> for IriRef<&'a str> {
    #[inline]
    fn from(iri: &'a IriRef<Cow<'a, str>>) -> Self {
        Self {
            iri: &iri.iri,
            positions: iri.positions,
        }
    }
}

/// Serializes as the plain string form (`self.iri`'s own `Serialize` impl), with no wrapping
/// object — exactly `oxiri`'s `IriRef<T>` shape, so a `&str`-backed value serializes as a
/// borrowed string and a `String`-backed value as an owned one, whichever `T` itself produces.
#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for IriRef<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.iri.serialize(serializer)
    }
}

/// Deserializes `T` (a string) and re-validates it via [`IriRef::parse`], turning a parse
/// failure into a [`serde::de::Error::custom`] — exactly `oxiri`'s `IriRef<T>` shape.
#[cfg(feature = "serde")]
impl<'de, T: Deref<Target = str> + Deserialize<'de>> Deserialize<'de> for IriRef<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        Self::parse(T::deserialize(deserializer)?).map_err(Error::custom)
    }
}
