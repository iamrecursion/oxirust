//! [`Iri`]: a validated, guaranteed-absolute RFC 3987 IRI.

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::str::FromStr;

use crate::error::{IriParseError, IriParseErrorKind, IriRelativizeError};
use crate::iri_ref::IriRef;
use crate::positions::{IriElementsPositions, find_iri_positions, has_dot_segment};
use crate::resolve::OutputBuffer;
use crate::validate::validate_iri;

/// A [RFC 3987](https://www.rfc-editor.org/rfc/rfc3987) `IRI` (as opposed to `IRI-reference`).
///
/// Unlike [`crate::IriRef`], instances of this type are guaranteed absolute: they always have a
/// `scheme`. The type is generic over its text storage `T` — anything that dereferences to
/// [`str`], typically `&str` for a zero-copy borrow or `String`/`Box<str>` for an owned value.
///
/// # Examples
///
/// ```
/// use oxixml_iri::{Iri, IriRef};
///
/// // Parse and validate a base IRI.
/// let base = Iri::parse("http://example.com/bar/baz")?;
///
/// // Resolve a relative reference against it.
/// let resolved = base.resolve(&IriRef::parse("bat#foo")?)?;
/// assert_eq!(resolved.as_str(), "http://example.com/bar/bat#foo");
///
/// // `Iri::parse` rejects relative references: every `Iri` has a scheme.
/// assert!(Iri::parse("../bar/baz").is_err());
///
/// // Relativizing is the (partial) inverse of resolving.
/// let relative = base.relativize(&Iri::parse("http://example.com/bar/bat#foo")?)?;
/// assert_eq!(relative, "bat#foo");
/// assert_eq!(base.resolve(&relative)?, resolved);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy)]
pub struct Iri<T>(IriRef<T>);

impl<T: Deref<Target = str>> Iri<T> {
    /// Parses and validates `iri` as an `IRI` following the grammar in
    /// [RFC 3987](https://www.rfc-editor.org/rfc/rfc3987).
    ///
    /// This takes ownership of `iri` (or borrows it, for `T = &str`) without any further
    /// allocation.
    ///
    /// Use [`parse_unchecked`](Self::parse_unchecked) instead if `iri` is already known to be
    /// valid, to skip the validation pass.
    ///
    /// # Errors
    ///
    /// Returns [`IriParseError`] if `iri` does not match the `IRI` grammar — including if it
    /// has no `scheme` (an `IRI`, unlike an `IRI-reference`, must be absolute).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// Iri::parse("http://example.com/bar/baz")?;
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    pub fn parse(iri: T) -> Result<Self, IriParseError> {
        let iri = Self::parse_unchecked(iri);
        validate_iri(&iri)?;
        Ok(iri)
    }

    /// Builds an `Iri` from `iri` without validating it.
    ///
    /// See the caveats on [`IriRef::parse_unchecked`](crate::IriRef::parse_unchecked): only use
    /// this when `iri` is already known-valid and absolute.
    ///
    /// # Panics
    ///
    /// Not on this call itself, which always succeeds -- but every `Iri` is assumed to be
    /// absolute, so [`scheme`](Self::scheme) unconditionally `.expect()`s one. If `iri` turns
    /// out not to have a scheme (or otherwise isn't a valid absolute IRI), the value it
    /// produces here is well-formed enough to *store*, but calling `scheme()` on it -- or
    /// anything else relying on absoluteness -- panics later, possibly far from this call site.
    /// Use [`parse`](Self::parse) instead unless `iri` is already known-valid and absolute.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// Iri::parse_unchecked("http://example.com/bar/baz");
    /// ```
    #[must_use]
    pub fn parse_unchecked(iri: T) -> Self {
        let positions = find_iri_positions(iri.as_ref());
        Self(IriRef { iri, positions })
    }

    /// Returns the underlying [`IriRef`], borrowed — used internally by
    /// [`validate_iri`](crate::validate::validate_iri) to check absoluteness and access
    /// components *before* [`scheme`](Self::scheme)'s "always absolute" invariant has been
    /// established (calling `scheme()` on a not-yet-validated, possibly relative `Iri` would
    /// panic; this accessor never does).
    #[inline]
    pub(crate) fn as_iri_ref(&self) -> &IriRef<T> {
        &self.0
    }

    /// Resolves `reference` against `self` as the base, following the strict
    /// [RFC 3986 §5](https://www.rfc-editor.org/rfc/rfc3986#section-5) relative-resolution
    /// algorithm, and validates the result.
    ///
    /// Since `self` is already absolute, the result is always itself an absolute `Iri`.
    ///
    /// Use [`resolve_unchecked`](Self::resolve_unchecked) instead if both `self` and
    /// `reference` are already known-valid, to skip the validation pass.
    ///
    /// # Errors
    ///
    /// See [`IriRef::resolve`](crate::IriRef::resolve).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::{Iri, IriRef};
    ///
    /// let base = Iri::parse("http://example.com/bar/baz")?;
    /// let resolved = base.resolve(&IriRef::parse("bat#foo")?)?;
    /// assert_eq!(resolved.into_inner(), "http://example.com/bar/bat#foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    pub fn resolve<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
    ) -> Result<Iri<String>, IriParseError> {
        Ok(Iri(self.0.resolve(reference)?))
    }

    /// Variant of [`resolve`](Self::resolve) that skips validating the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::{Iri, IriRef};
    ///
    /// let base = Iri::parse_unchecked("http://example.com/bar/baz");
    /// let resolved = base.resolve_unchecked(&IriRef::parse_unchecked("bat#foo"));
    /// assert_eq!(resolved.into_inner(), "http://example.com/bar/bat#foo");
    /// ```
    #[must_use]
    pub fn resolve_unchecked<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
    ) -> Iri<String> {
        Iri(self.0.resolve_unchecked(reference))
    }

    /// Variant of [`resolve`](Self::resolve) that writes into `output_buffer` instead of
    /// allocating a fresh [`String`].
    ///
    /// # Errors
    ///
    /// See [`IriRef::resolve`](crate::IriRef::resolve).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::{Iri, IriRef};
    ///
    /// let base = Iri::parse("http://example.com/bar/baz")?;
    /// let mut buffer = String::new();
    /// base.resolve_into(&IriRef::parse("bat#foo")?, &mut buffer)?;
    /// assert_eq!(buffer, "http://example.com/bar/bat#foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    pub fn resolve_into<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
        output_buffer: &mut impl OutputBuffer,
    ) -> Result<(), IriParseError> {
        self.0.resolve_into(reference, output_buffer)
    }

    /// Variant of [`resolve_into`](Self::resolve_into) that skips validating the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::{Iri, IriRef};
    ///
    /// let base = Iri::parse_unchecked("http://example.com/bar/baz");
    /// let mut buffer = String::new();
    /// base.resolve_into_unchecked(&IriRef::parse_unchecked("bat#foo"), &mut buffer);
    /// assert_eq!(buffer, "http://example.com/bar/bat#foo");
    /// ```
    pub fn resolve_into_unchecked<T2: Deref<Target = str>>(
        &self,
        reference: &IriRef<T2>,
        output_buffer: &mut impl OutputBuffer,
    ) {
        self.0.resolve_into_unchecked(reference, output_buffer);
    }

    /// Returns the shortest `IriRef` that, when [`resolve`](Self::resolve)d against `self`,
    /// yields `target` back.
    ///
    /// This is the (partial) inverse of [`resolve`](Self::resolve): an equal target yields a
    /// same-document reference (`""`, or just the fragment); a target sharing `self`'s scheme
    /// and authority yields a relative path (using `../` climbs when a common path prefix
    /// exists); a target with a different scheme or authority falls back to the full,
    /// unmodified `target` string.
    ///
    /// The output is deliberately the *simplest correct* answer this crate can compute, not
    /// necessarily the theoretically shortest one; it may change between minor versions as the
    /// heuristic improves, but it always round-trips: `base.resolve(&base.relativize(target)?)`
    /// always equals `target`.
    ///
    /// # Errors
    ///
    /// Returns [`IriRelativizeError`] if `target`'s path contains a `.` or `..` segment: the
    /// resolution algorithm consumes those while merging, so no relative reference could
    /// resolve back to a path containing one.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let base = Iri::parse("http://example.com/bar/baz")?;
    /// let target = Iri::parse("http://example.com/bar/bat#foo")?;
    /// let relative = base.relativize(&target)?;
    /// assert_eq!(relative, "bat#foo");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn relativize<T2: Deref<Target = str>>(
        &self,
        target: &Iri<T2>,
    ) -> Result<IriRef<String>, IriRelativizeError> {
        relativize(self, target)
    }

    /// Returns an `Iri<&str>` borrowing this IRI's text.
    #[inline]
    #[must_use]
    pub fn as_ref(&self) -> Iri<&str> {
        Iri(self.0.as_ref())
    }

    /// Returns the underlying textual representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let iri = Iri::parse("http://example.com/foo")?;
    /// assert_eq!(iri.as_str(), "http://example.com/foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Consumes `self`, returning the underlying textual representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let iri = Iri::parse("http://example.com/foo")?;
    /// assert_eq!(iri.into_inner(), "http://example.com/foo");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }

    /// Returns the `scheme` component.
    ///
    /// Always present (an `Iri` is always absolute); use a case-insensitive comparison to test
    /// against a specific scheme, since scheme names are case-insensitive
    /// ([RFC 3986 §3.1](https://www.rfc-editor.org/rfc/rfc3986#section-3.1)).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let iri = Iri::parse("hTTp://example.com")?;
    /// assert_eq!(iri.scheme(), "hTTp");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn scheme(&self) -> &str {
        self.0
            .scheme()
            .expect("an Iri is always absolute and therefore always has a scheme")
    }

    /// Returns the `authority` component, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let http = Iri::parse("http://user:pass@example.com:80/my/path")?;
    /// assert_eq!(http.authority(), Some("user:pass@example.com:80"));
    ///
    /// let mailto = Iri::parse("mailto:foo@bar.com")?;
    /// assert_eq!(mailto.authority(), None);
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn authority(&self) -> Option<&str> {
        self.0.authority()
    }

    /// Returns the `path` component. Always present, though it may be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let http = Iri::parse("http://user:pass@example.com:80/my/path?foo=bar")?;
    /// assert_eq!(http.path(), "/my/path");
    ///
    /// let mailto = Iri::parse("mailto:foo@bar.com")?;
    /// assert_eq!(mailto.path(), "foo@bar.com");
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn path(&self) -> &str {
        self.0.path()
    }

    /// Returns the `query` component, if present (excluding the leading `?`).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let iri = Iri::parse("http://example.com/my/path?query=foo#frag")?;
    /// assert_eq!(iri.query(), Some("query=foo"));
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn query(&self) -> Option<&str> {
        self.0.query()
    }

    /// Returns the `fragment` component, if present (excluding the leading `#`).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_iri::Iri;
    ///
    /// let iri = Iri::parse("http://example.com/my/path?query=foo#frag")?;
    /// assert_eq!(iri.fragment(), Some("frag"));
    /// # Ok::<(), oxixml_iri::IriParseError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn fragment(&self) -> Option<&str> {
        self.0.fragment()
    }
}

/// The actual `relativize` implementation, kept as a free function (rather than inlined into
/// the `impl` block) purely for readability; see [`Iri::relativize`] for the contract.
fn relativize<T1: Deref<Target = str>, T2: Deref<Target = str>>(
    base: &Iri<T1>,
    target: &Iri<T2>,
) -> Result<IriRef<String>, IriRelativizeError> {
    let target_authority = target.authority();
    let base_authority = base.authority();
    let target_path = target.path();
    let base_path = base.path();
    let target_query = target.query();
    let base_query = base.query();

    // The resolution algorithm consumes `.`/`..` segments while merging; a relative reference
    // could never resolve back to a target path that has one.
    if has_dot_segment(target_path) {
        return Err(IriRelativizeError);
    }

    if target.scheme() != base.scheme()
        || target_authority.is_none()
            && (base_authority.is_some()
                || target_path.is_empty()
                    && (!base_path.is_empty() || target_query.is_none() && base_query.is_some()))
        || target_path
            // A leading path segment containing no `/` could be confused with a scheme.
            .split_once(':')
            .is_some_and(|(candidate_scheme, _)| !candidate_scheme.contains('/'))
    {
        return Ok(IriRef {
            iri: target.0.as_str().to_string(),
            positions: target.0.positions,
        });
    }
    if target_authority != base_authority
        // The resolution algorithm does not handle empty paths against a non-empty base:
        || target_path.is_empty() && (!base_path.is_empty() || base_query.is_some())
        // A path starting with `//` right after the (elided) scheme would be re-parsed as an
        // authority.
        || target_path.starts_with("//")
    {
        return Ok(IriRef {
            iri: target.0.as_str()[target.0.positions.scheme_end..].to_string(),
            positions: IriElementsPositions {
                scheme_end: 0,
                authority_end: target.0.positions.authority_end - target.0.positions.scheme_end,
                path_end: target.0.positions.path_end - target.0.positions.scheme_end,
                query_end: target.0.positions.query_end - target.0.positions.scheme_end,
            },
        });
    }
    if target_path != base_path || target_query.is_none() && base_query.is_some() {
        let shared_len = target_path
            .bytes()
            .zip(base_path.bytes())
            .take_while(|(l, r)| l == r)
            .count();
        // Back off to the last shared `/` boundary: a shared prefix that splits a path segment
        // in half is not a usable common prefix.
        let shared_len = target_path.as_bytes()[..shared_len]
            .iter()
            .rposition(|c| *c == b'/')
            .map_or(0, |n| n + 1);
        return if target_path[shared_len..].starts_with('/')
            || base_path[shared_len..].contains('/')
            || target_path[shared_len..].contains(':')
        {
            // The remainder still contains a `/` or a `:` that could be misread as an
            // authority or a scheme once the leading `../` climbs are stripped away — fall
            // back to a full authority-relative or complete reference.
            if !target_path.starts_with('/') && !base_path.is_empty() {
                Ok(IriRef {
                    iri: target.0.as_str().to_string(),
                    positions: target.0.positions,
                })
            } else {
                Ok(IriRef {
                    iri: target.0.as_str()[target.0.positions.authority_end..].to_string(),
                    positions: IriElementsPositions {
                        scheme_end: 0,
                        authority_end: 0,
                        path_end: target.0.positions.path_end - target.0.positions.authority_end,
                        query_end: target.0.positions.query_end - target.0.positions.authority_end,
                    },
                })
            }
        } else if target_path[shared_len..].is_empty() {
            // The whole of the target path is shared with the base path (up to the file
            // name): a same-directory reference of exactly ".".
            let mut iri = String::with_capacity(
                target.0.as_str().len() - target.0.positions.authority_end - shared_len + 1,
            );
            iri.push('.');
            iri.push_str(&target.0.as_str()[target.0.positions.authority_end + shared_len..]);
            Ok(IriRef {
                iri,
                positions: IriElementsPositions {
                    scheme_end: 0,
                    authority_end: 0,
                    path_end: target.0.positions.path_end
                        - target.0.positions.authority_end
                        - shared_len
                        + 1,
                    query_end: target.0.positions.query_end
                        - target.0.positions.authority_end
                        - shared_len
                        + 1,
                },
            })
        } else {
            // Only the last path segment differs: emit just that.
            Ok(IriRef {
                iri: target.0.as_str()[target.0.positions.authority_end + shared_len..].to_string(),
                positions: IriElementsPositions {
                    scheme_end: 0,
                    authority_end: 0,
                    path_end: target.0.positions.path_end
                        - target.0.positions.authority_end
                        - shared_len,
                    query_end: target.0.positions.query_end
                        - target.0.positions.authority_end
                        - shared_len,
                },
            })
        };
    }
    if target_query != base_query {
        return Ok(IriRef {
            iri: target.0.as_str()[target.0.positions.path_end..].to_string(),
            positions: IriElementsPositions {
                scheme_end: 0,
                authority_end: 0,
                path_end: 0,
                query_end: target.0.positions.query_end - target.0.positions.path_end,
            },
        });
    }
    Ok(IriRef {
        iri: target.0.as_str()[target.0.positions.query_end..].to_string(),
        positions: IriElementsPositions {
            scheme_end: 0,
            authority_end: 0,
            path_end: 0,
            query_end: 0,
        },
    })
}

impl<Lft: PartialEq<Rhs>, Rhs> PartialEq<Iri<Rhs>> for Iri<Lft> {
    #[inline]
    fn eq(&self, other: &Iri<Rhs>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<Lft: PartialEq<Rhs>, Rhs> PartialEq<IriRef<Rhs>> for Iri<Lft> {
    #[inline]
    fn eq(&self, other: &IriRef<Rhs>) -> bool {
        self.0.eq(other)
    }
}

impl<Lft: PartialEq<Rhs>, Rhs> PartialEq<Iri<Rhs>> for IriRef<Lft> {
    #[inline]
    fn eq(&self, other: &Iri<Rhs>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<str>> PartialEq<str> for Iri<T> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.0.eq(other)
    }
}

impl<'a, T: PartialEq<&'a str>> PartialEq<&'a str> for Iri<T> {
    #[inline]
    fn eq(&self, other: &&'a str) -> bool {
        self.0.eq(other)
    }
}

impl<T: PartialEq<String>> PartialEq<String> for Iri<T> {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.0.eq(other)
    }
}

impl<'a, T: PartialEq<Cow<'a, str>>> PartialEq<Cow<'a, str>> for Iri<T> {
    #[inline]
    fn eq(&self, other: &Cow<'a, str>) -> bool {
        self.0.eq(other)
    }
}

impl<T: PartialEq<str>> PartialEq<Iri<T>> for str {
    #[inline]
    fn eq(&self, other: &Iri<T>) -> bool {
        self.eq(&other.0)
    }
}

impl<'a, T: PartialEq<&'a str>> PartialEq<Iri<T>> for &'a str {
    #[inline]
    fn eq(&self, other: &Iri<T>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<String>> PartialEq<Iri<T>> for String {
    #[inline]
    fn eq(&self, other: &Iri<T>) -> bool {
        self.eq(&other.0)
    }
}

impl<'a, T: PartialEq<Cow<'a, str>>> PartialEq<Iri<T>> for Cow<'a, str> {
    #[inline]
    fn eq(&self, other: &Iri<T>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: Eq> Eq for Iri<T> {}

impl<T: Hash> Hash for Iri<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<Lft: PartialOrd<Rhs>, Rhs> PartialOrd<Iri<Rhs>> for Iri<Lft> {
    #[inline]
    fn partial_cmp(&self, other: &Iri<Rhs>) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<Lft: PartialOrd<Rhs>, Rhs> PartialOrd<IriRef<Rhs>> for Iri<Lft> {
    #[inline]
    fn partial_cmp(&self, other: &IriRef<Rhs>) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl<Lft: PartialOrd<Rhs>, Rhs> PartialOrd<Iri<Rhs>> for IriRef<Lft> {
    #[inline]
    fn partial_cmp(&self, other: &Iri<Rhs>) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}

impl<T: Ord> Ord for Iri<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: Deref<Target = str>> Deref for Iri<T> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.0.deref()
    }
}

impl<T: AsRef<str>> AsRef<str> for Iri<T> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl<T: Borrow<str>> Borrow<str> for Iri<T> {
    #[inline]
    fn borrow(&self) -> &str {
        self.0.borrow()
    }
}

impl<T: fmt::Debug> fmt::Debug for Iri<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: fmt::Display> fmt::Display for Iri<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for Iri<String> {
    type Err = IriParseError;

    #[inline]
    fn from_str(iri: &str) -> Result<Self, IriParseError> {
        Self::parse(iri.to_owned())
    }
}

impl<'a> From<Iri<&'a str>> for Iri<String> {
    #[inline]
    fn from(iri: Iri<&'a str>) -> Self {
        Self(iri.0.into())
    }
}

impl<'a> From<Iri<Cow<'a, str>>> for Iri<String> {
    #[inline]
    fn from(iri: Iri<Cow<'a, str>>) -> Self {
        Self(iri.0.into())
    }
}

impl From<Iri<Box<str>>> for Iri<String> {
    #[inline]
    fn from(iri: Iri<Box<str>>) -> Self {
        Self(iri.0.into())
    }
}

impl<'a> From<Iri<&'a str>> for Iri<Cow<'a, str>> {
    #[inline]
    fn from(iri: Iri<&'a str>) -> Self {
        Self(iri.0.into())
    }
}

impl From<Iri<String>> for Iri<Cow<'_, str>> {
    #[inline]
    fn from(iri: Iri<String>) -> Self {
        Self(iri.0.into())
    }
}

impl<'a> From<&'a Iri<String>> for Iri<&'a str> {
    #[inline]
    fn from(iri: &'a Iri<String>) -> Self {
        Self(iri.0.as_ref())
    }
}

impl<'a> From<&'a Iri<Cow<'a, str>>> for Iri<&'a str> {
    #[inline]
    fn from(iri: &'a Iri<Cow<'a, str>>) -> Self {
        Self(iri.0.as_ref())
    }
}

impl<T: Deref<Target = str>> From<Iri<T>> for IriRef<T> {
    #[inline]
    fn from(iri: Iri<T>) -> Self {
        iri.0
    }
}

impl<T: Deref<Target = str>> TryFrom<IriRef<T>> for Iri<T> {
    type Error = IriParseError;

    fn try_from(iri: IriRef<T>) -> Result<Self, IriParseError> {
        if iri.is_absolute() {
            Ok(Self(iri))
        } else {
            Err(IriParseErrorKind::NoScheme.into())
        }
    }
}

/// Serializes as the plain string form, delegating to the wrapped [`IriRef`]'s own `Serialize`
/// impl — exactly `oxiri`'s `Iri<T>` shape (no `{"scheme": ..., ...}` wrapping object).
#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for Iri<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

/// Deserializes through [`IriRef`]'s [`Deserialize`] impl, then re-validates absoluteness via
/// `TryFrom<IriRef<T>>`, turning either failure into a [`serde::de::Error::custom`] — exactly
/// `oxiri`'s `Iri<T>` shape.
#[cfg(feature = "serde")]
impl<'de, T: Deref<Target = str> + Deserialize<'de>> Deserialize<'de> for Iri<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        IriRef::deserialize(deserializer)?
            .try_into()
            .map_err(Error::custom)
    }
}
