//! [`QName`]: a lexical, unresolved qualified name.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::str::FromStr;

use crate::error::QNameParseError;
use crate::validate::{is_ncname, split_qname};

/// A [`QName`](https://www.w3.org/TR/xml-names/#NT-QName): a validated `prefix:local` or bare
/// `local` name, as written in an XML document — *before* the prefix is resolved against a
/// namespace context.
///
/// `QName` is generic over its text storage `T`, mirroring the sibling `oxixml-iri` crate's
/// `Iri<T>`: pass `T = &str` for a zero-copy borrow over a buffer you already own (typical
/// inside a streaming parser), or `T = String`/`Box<str>` for an owned value. Either way there
/// is exactly one allocation (or none, for a borrow): `QName` stores the whole `value` string as
/// given and remembers only the byte offset of the separating colon, rather than allocating
/// separate `prefix`/`local` buffers.
///
/// A `QName` on its own carries no namespace information — `book:title` and `xhtml:title` are
/// unrelated as `QName`s even if a [`crate::NamespaceStack`] happens to bind both prefixes to
/// the same namespace name. Use [`crate::NamespaceStack::resolve_qname`] to turn a `QName` into
/// a namespace-aware [`crate::ExpandedName`].
///
/// # Examples
///
/// ```
/// use oxixml_qname::QName;
///
/// // Borrowed, zero-copy.
/// let borrowed: QName<&str> = QName::parse("book:title")?;
/// assert_eq!(borrowed.prefix(), Some("book"));
/// assert_eq!(borrowed.local_part(), "title");
/// assert_eq!(borrowed.as_str(), "book:title");
///
/// // Owned, via `FromStr`.
/// let owned: QName<String> = "title".parse()?;
/// assert_eq!(owned.prefix(), None);
/// assert_eq!(owned.local_part(), "title");
///
/// // Borrowed and owned `QName`s compare equal by their text.
/// assert_eq!(borrowed, QName::parse("book:title".to_owned())?);
/// # Ok::<(), oxixml_qname::QNameParseError>(())
/// ```
#[derive(Clone, Copy)]
pub struct QName<T> {
    value: T,
    colon: Option<usize>,
}

impl<T: Deref<Target = str>> QName<T> {
    /// Parses and validates `value` as a `QName`, per the
    /// [`QName` production](https://www.w3.org/TR/xml-names/#NT-QName).
    ///
    /// This takes ownership of `value` (or borrows it, for `T = &str`) without any further
    /// allocation; the returned `QName` simply remembers the byte offset of the colon, if any.
    ///
    /// Use [`parse_unchecked`](Self::parse_unchecked) instead if `value` is already known to be
    /// a valid `QName`, to skip the validation pass.
    ///
    /// # Errors
    ///
    /// Returns [`QNameParseError`] if `value` is empty, has more than one colon, or has a
    /// prefix or local part that is not a valid `NCName`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::QName;
    ///
    /// assert!(QName::parse("book:title").is_ok());
    /// assert!(QName::parse("title").is_ok());
    /// assert!(QName::parse("").is_err());
    /// assert!(QName::parse(":title").is_err());
    /// assert!(QName::parse("book:").is_err());
    /// assert!(QName::parse("book:title:extra").is_err());
    /// ```
    pub fn parse(value: T) -> Result<Self, QNameParseError> {
        let colon = match split_qname(value.deref()) {
            None => return Err(QNameParseError::empty()),
            Some((Some(prefix), local)) => {
                if !is_ncname(prefix) {
                    return Err(QNameParseError::invalid_prefix());
                }
                if !is_ncname(local) {
                    return Err(QNameParseError::invalid_local_part());
                }
                Some(prefix.len())
            }
            Some((None, local)) => {
                if !is_ncname(local) {
                    return Err(QNameParseError::invalid_local_part());
                }
                None
            }
        };
        Ok(Self { value, colon })
    }

    /// Builds a `QName` from `value` without validating it.
    ///
    /// The [`prefix`](Self::prefix) and [`local_part`](Self::local_part) accessors still behave
    /// correctly (they split on the first colon, exactly as [`parse`](Self::parse) does), but no
    /// `NCName` character-class checking is performed — only use this when `value` is already
    /// known-valid, since the result of calling other methods on an invalid `QName` is
    /// unspecified (though never undefined behavior; this crate has no `unsafe` code).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::QName;
    ///
    /// let qname = QName::parse_unchecked("book:title");
    /// assert_eq!(qname.prefix(), Some("book"));
    /// ```
    #[must_use]
    pub fn parse_unchecked(value: T) -> Self {
        let colon = value.deref().find(':');
        Self { value, colon }
    }

    /// The prefix, if any (the text before the colon).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::QName;
    ///
    /// assert_eq!(QName::parse("book:title")?.prefix(), Some("book"));
    /// assert_eq!(QName::parse("title")?.prefix(), None);
    /// # Ok::<(), oxixml_qname::QNameParseError>(())
    /// ```
    #[must_use]
    pub fn prefix(&self) -> Option<&str> {
        self.colon.map(|idx| &self.value.deref()[..idx])
    }

    /// The local part (the text after the colon, or the whole value if there is no colon).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::QName;
    ///
    /// assert_eq!(QName::parse("book:title")?.local_part(), "title");
    /// assert_eq!(QName::parse("title")?.local_part(), "title");
    /// # Ok::<(), oxixml_qname::QNameParseError>(())
    /// ```
    #[must_use]
    pub fn local_part(&self) -> &str {
        match self.colon {
            Some(idx) => &self.value.deref()[idx + 1..],
            None => self.value.deref(),
        }
    }

    /// Returns `true` if this `QName` has a prefix (was written as `prefix:local`).
    #[must_use]
    pub fn is_prefixed(&self) -> bool {
        self.colon.is_some()
    }

    /// The full lexical text of this `QName`, e.g. `"book:title"` or `"title"`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.value.deref()
    }

    /// Consumes this `QName`, returning the underlying storage `T`.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Returns a zero-copy `QName<&str>` borrowing from this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::QName;
    ///
    /// let owned: QName<String> = "book:title".parse()?;
    /// let borrowed: QName<&str> = owned.as_borrowed();
    /// assert_eq!(borrowed.prefix(), Some("book"));
    /// # Ok::<(), oxixml_qname::QNameParseError>(())
    /// ```
    #[must_use]
    pub fn as_borrowed(&self) -> QName<&str> {
        QName {
            value: self.value.deref(),
            colon: self.colon,
        }
    }

    /// The byte offset of the separating colon within [`as_str`](Self::as_str), if this `QName`
    /// is prefixed.
    ///
    /// This is deliberately `pub(crate)`, not part of the public API: it exists only so that
    /// callers elsewhere in this crate (namely
    /// [`NamespaceStack::resolve_qname_ref`](crate::NamespaceStack::resolve_qname_ref)) can
    /// re-slice an *original* `&'a str` they still hold by this offset, obtaining `&'a str`
    /// components. Slicing through [`prefix`](Self::prefix)/[`local_part`](Self::local_part)
    /// instead would tie the result to the lifetime of the borrow of `self` used to call them
    /// (per `Deref`'s signature), which is too short for that use case even when `T = &'a str`.
    pub(crate) fn colon_offset(&self) -> Option<usize> {
        self.colon
    }
}

impl<T: Deref<Target = str>> Deref for QName<T> {
    type Target = str;

    fn deref(&self) -> &str {
        self.value.deref()
    }
}

impl<T: AsRef<str>> AsRef<str> for QName<T> {
    fn as_ref(&self) -> &str {
        self.value.as_ref()
    }
}

impl<T: Borrow<str>> Borrow<str> for QName<T> {
    fn borrow(&self) -> &str {
        self.value.borrow()
    }
}

impl<T: Deref<Target = str> + fmt::Debug> fmt::Debug for QName<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QName")
            .field("prefix", &self.prefix())
            .field("local_part", &self.local_part())
            .finish()
    }
}

impl<T: fmt::Display> fmt::Display for QName<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl<Lhs: Deref<Target = str>, Rhs: Deref<Target = str>> PartialEq<QName<Rhs>> for QName<Lhs> {
    fn eq(&self, other: &QName<Rhs>) -> bool {
        self.value.deref() == other.value.deref()
    }
}

impl<T: Deref<Target = str>> Eq for QName<T> {}

impl<T: Deref<Target = str>> Hash for QName<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.deref().hash(state);
    }
}

impl<Lhs: Deref<Target = str>, Rhs: Deref<Target = str>> PartialOrd<QName<Rhs>> for QName<Lhs> {
    fn partial_cmp(&self, other: &QName<Rhs>) -> Option<Ordering> {
        Some(self.value.deref().cmp(other.value.deref()))
    }
}

impl<T: Deref<Target = str>> Ord for QName<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.deref().cmp(other.value.deref())
    }
}

impl FromStr for QName<String> {
    type Err = QNameParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        QName::parse(s.to_owned())
    }
}

impl<'a> From<QName<&'a str>> for QName<String> {
    fn from(qname: QName<&'a str>) -> Self {
        QName {
            value: qname.value.to_owned(),
            colon: qname.colon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn parse_borrowed_unprefixed() -> Result<(), QNameParseError> {
        let qname = QName::parse("title")?;
        assert_eq!(qname.prefix(), None);
        assert_eq!(qname.local_part(), "title");
        assert!(!qname.is_prefixed());
        assert_eq!(qname.as_str(), "title");
        Ok(())
    }

    #[test]
    fn parse_borrowed_prefixed() -> Result<(), QNameParseError> {
        let qname = QName::parse("book:title")?;
        assert_eq!(qname.prefix(), Some("book"));
        assert_eq!(qname.local_part(), "title");
        assert!(qname.is_prefixed());
        Ok(())
    }

    #[test]
    fn parse_rejects_malformed_input() {
        assert!(QName::parse("").is_err());
        assert!(QName::parse(":title").is_err());
        assert!(QName::parse("book:").is_err());
        assert!(QName::parse("book:title:extra").is_err());
        assert!(QName::parse("0book:title").is_err());
        assert!(QName::parse("book:0title").is_err());
    }

    #[test]
    fn from_str_produces_owned_qname() -> Result<(), QNameParseError> {
        let qname: QName<String> = "book:title".parse()?;
        assert_eq!(qname.prefix(), Some("book"));
        assert_eq!(qname.local_part(), "title");
        Ok(())
    }

    #[test]
    fn borrowed_and_owned_compare_and_hash_equal() -> Result<(), QNameParseError> {
        let borrowed = QName::parse("book:title")?;
        let owned: QName<String> = "book:title".parse()?;
        assert_eq!(borrowed, owned);
        assert_eq!(owned, borrowed);

        // `Hash` agrees with `Eq`: a `QName<String>` built from the same text is found in a set
        // keyed by another `QName<String>` with equal text.
        let mut set = HashSet::new();
        set.insert(QName::parse("book:title".to_owned())?);
        assert!(set.contains(&owned));
        Ok(())
    }

    #[test]
    fn ordering_is_lexicographic_over_the_full_text() -> Result<(), QNameParseError> {
        let a = QName::parse("a:z")?;
        let b = QName::parse("b:a")?;
        assert!(a < b); // "a:z" < "b:a" lexicographically
        Ok(())
    }

    #[test]
    fn as_borrowed_roundtrips() -> Result<(), QNameParseError> {
        let owned: QName<String> = "book:title".parse()?;
        let borrowed = owned.as_borrowed();
        assert_eq!(borrowed.prefix(), Some("book"));
        assert_eq!(borrowed.local_part(), "title");
        Ok(())
    }

    #[test]
    fn into_qname_string_conversion() -> Result<(), QNameParseError> {
        let borrowed = QName::parse("book:title")?;
        let owned: QName<String> = borrowed.into();
        assert_eq!(owned.as_str(), "book:title");
        Ok(())
    }

    #[test]
    fn parse_unchecked_still_splits_correctly() {
        let qname = QName::parse_unchecked("book:title");
        assert_eq!(qname.prefix(), Some("book"));
        assert_eq!(qname.local_part(), "title");
    }

    #[test]
    fn debug_shows_prefix_and_local_part() -> Result<(), QNameParseError> {
        let qname = QName::parse("book:title")?;
        let debug = format!("{qname:?}");
        assert!(debug.contains("book"));
        assert!(debug.contains("title"));
        Ok(())
    }

    #[test]
    fn display_shows_full_text() -> Result<(), QNameParseError> {
        let qname = QName::parse("book:title")?;
        assert_eq!(qname.to_string(), "book:title");
        Ok(())
    }
}
