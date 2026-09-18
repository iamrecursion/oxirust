//! Trait implementations for [`LanguageTag`]: comparison, hashing, string conversions, and
//! cross-storage `From` conversions.
//!
//! Equality, ordering, and hashing all delegate directly to the backing storage's own `str`
//! comparison — i.e. they are **case-sensitive**, exactly like comparing the two tag strings
//! with `==`. `LanguageTag::parse("en-US") != LanguageTag::parse("en-us")` even though both are
//! well-formed tags for the same locale; callers that want locale-equivalence should compare
//! [`LanguageTag::parse_and_normalize`] output instead, or fold case themselves.

use alloc::borrow::{Cow, ToOwned};
use alloc::boxed::Box;
use alloc::string::String;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::str::FromStr;

use crate::error::LanguageTagParseError;
use crate::tag::LanguageTag;

impl<Lhs, Rhs> PartialEq<LanguageTag<Rhs>> for LanguageTag<Lhs>
where
    Lhs: PartialEq<Rhs>,
{
    #[inline]
    fn eq(&self, other: &LanguageTag<Rhs>) -> bool {
        self.tag == other.tag
    }
}

impl<T> PartialEq<str> for LanguageTag<T>
where
    T: PartialEq<str>,
{
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.tag == *other
    }
}

impl<T> PartialEq<LanguageTag<T>> for str
where
    T: PartialEq<str>,
{
    #[inline]
    fn eq(&self, other: &LanguageTag<T>) -> bool {
        other.tag == *self
    }
}

impl<'a, T> PartialEq<&'a str> for LanguageTag<T>
where
    T: PartialEq<&'a str>,
{
    #[inline]
    fn eq(&self, other: &&'a str) -> bool {
        self.tag == *other
    }
}

impl<'a, T> PartialEq<LanguageTag<T>> for &'a str
where
    T: PartialEq<&'a str>,
{
    #[inline]
    fn eq(&self, other: &LanguageTag<T>) -> bool {
        other.tag == *self
    }
}

impl<T> PartialEq<String> for LanguageTag<T>
where
    T: PartialEq<String>,
{
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.tag == *other
    }
}

impl<T> PartialEq<LanguageTag<T>> for String
where
    T: PartialEq<String>,
{
    #[inline]
    fn eq(&self, other: &LanguageTag<T>) -> bool {
        other.tag == *self
    }
}

impl<'a, T> PartialEq<Cow<'a, str>> for LanguageTag<T>
where
    T: PartialEq<Cow<'a, str>>,
{
    #[inline]
    fn eq(&self, other: &Cow<'a, str>) -> bool {
        self.tag == *other
    }
}

impl<'a, T> PartialEq<LanguageTag<T>> for Cow<'a, str>
where
    T: PartialEq<Cow<'a, str>>,
{
    #[inline]
    fn eq(&self, other: &LanguageTag<T>) -> bool {
        other.tag == *self
    }
}

impl<T: Eq> Eq for LanguageTag<T> {}

impl<T: Hash> Hash for LanguageTag<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag.hash(state);
    }
}

impl<T: PartialOrd> PartialOrd for LanguageTag<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.tag.partial_cmp(&other.tag)
    }
}

impl<T: Ord> Ord for LanguageTag<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag.cmp(&other.tag)
    }
}

impl<T: Deref<Target = str>> Deref for LanguageTag<T> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.tag.deref()
    }
}

impl<T: AsRef<str>> AsRef<str> for LanguageTag<T> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.tag.as_ref()
    }
}

impl<T: Borrow<str>> Borrow<str> for LanguageTag<T> {
    #[inline]
    fn borrow(&self) -> &str {
        self.tag.borrow()
    }
}

impl<T: fmt::Debug> fmt::Debug for LanguageTag<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LanguageTag").field(&self.tag).finish()
    }
}

impl<T: fmt::Display> fmt::Display for LanguageTag<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.tag, f)
    }
}

impl FromStr for LanguageTag<String> {
    type Err = LanguageTagParseError;

    /// Equivalent to [`LanguageTag::parse_and_normalize`].
    #[inline]
    fn from_str(tag: &str) -> Result<Self, Self::Err> {
        Self::parse_and_normalize(tag)
    }
}

impl<'a> From<LanguageTag<&'a str>> for LanguageTag<String> {
    #[inline]
    fn from(tag: LanguageTag<&'a str>) -> Self {
        Self {
            tag: tag.tag.to_owned(),
            positions: tag.positions,
        }
    }
}

impl<'a> From<LanguageTag<Cow<'a, str>>> for LanguageTag<String> {
    #[inline]
    fn from(tag: LanguageTag<Cow<'a, str>>) -> Self {
        Self {
            tag: tag.tag.into_owned(),
            positions: tag.positions,
        }
    }
}

impl From<LanguageTag<Box<str>>> for LanguageTag<String> {
    #[inline]
    fn from(tag: LanguageTag<Box<str>>) -> Self {
        Self {
            tag: tag.tag.into(),
            positions: tag.positions,
        }
    }
}

impl<'a> From<LanguageTag<&'a str>> for LanguageTag<Cow<'a, str>> {
    #[inline]
    fn from(tag: LanguageTag<&'a str>) -> Self {
        Self {
            tag: Cow::Borrowed(tag.tag),
            positions: tag.positions,
        }
    }
}

impl<'a> From<LanguageTag<String>> for LanguageTag<Cow<'a, str>> {
    #[inline]
    fn from(tag: LanguageTag<String>) -> Self {
        Self {
            tag: Cow::Owned(tag.tag),
            positions: tag.positions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::ToString;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(value: impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn equality_is_case_sensitive() -> Result<(), LanguageTagParseError> {
        let tag = LanguageTag::parse("en-fr")?;
        assert_eq!(tag, "en-fr");
        assert_ne!(tag, "en-FR");
        assert_eq!("en-fr", tag);
        assert_eq!(hash_of(tag), hash_of("en-fr"));
        assert_ne!(hash_of(tag), hash_of("en-FR"));
        Ok(())
    }

    #[test]
    fn cross_storage_equality() -> Result<(), LanguageTagParseError> {
        let borrowed = LanguageTag::parse("en-fr")?;
        let owned = LanguageTag::parse("en-fr".to_string())?;
        assert_eq!(borrowed, owned);
        Ok(())
    }

    #[test]
    fn ordering_matches_str_ordering() -> Result<(), LanguageTagParseError> {
        let a = LanguageTag::parse("de")?;
        let b = LanguageTag::parse("fr")?;
        assert!(a < b);
        Ok(())
    }

    #[test]
    fn deref_and_str_methods() -> Result<(), LanguageTagParseError> {
        let tag = LanguageTag::parse("en-fr")?;
        assert!(tag.starts_with("en-"));
        Ok(())
    }

    #[test]
    fn debug_and_display() -> Result<(), LanguageTagParseError> {
        let tag = LanguageTag::parse("en-fr")?;
        assert_eq!(tag.to_string(), "en-fr");
        assert_eq!(format!("{tag:?}"), "LanguageTag(\"en-fr\")");
        Ok(())
    }

    #[test]
    fn from_str_normalizes() -> Result<(), LanguageTagParseError> {
        let tag: LanguageTag<String> = "en-us".parse()?;
        assert_eq!(tag.as_str(), "en-US");
        Ok(())
    }

    #[test]
    fn from_conversions_preserve_content() -> Result<(), LanguageTagParseError> {
        let borrowed = LanguageTag::parse("en-us")?;
        let owned: LanguageTag<String> = borrowed.into();
        assert_eq!(owned.as_str(), "en-us");

        let borrowed = LanguageTag::parse("en-us")?;
        let cow: LanguageTag<Cow<'_, str>> = borrowed.into();
        assert_eq!(cow.as_str(), "en-us");
        let owned_from_cow: LanguageTag<String> = cow.into();
        assert_eq!(owned_from_cow.as_str(), "en-us");
        Ok(())
    }
}
