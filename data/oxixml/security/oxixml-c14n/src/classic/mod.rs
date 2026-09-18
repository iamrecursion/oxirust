//! Canonical XML 1.0, Canonical XML 1.1 and Exclusive XML Canonicalization 1.0.
//!
//! The three algorithms share one traversal and differ only in how they treat
//! the namespace axis and the `xml:*` attributes of an orphan element, so they
//! are implemented once and selected with [`Variant`].
//!
//! | | namespace axis | `xml:*` inheritance |
//! |---|---|---|
//! | [Canonical XML 1.0][c14n10] | the whole inherited context is rendered at the apex | every `xml:*` attribute of the omitted ancestors |
//! | [Canonical XML 1.1][c14n11] | same as 1.0 | only `xml:lang` and `xml:space`, plus `xml:base` fixup; never `xml:id` |
//! | [Exclusive C14N 1.0][exc] | only *visibly utilized* declarations, plus the `InclusiveNamespaces` `PrefixList` | none |
//!
//! [c14n10]: https://www.w3.org/TR/xml-c14n
//! [c14n11]: https://www.w3.org/TR/xml-c14n11/
//! [exc]: https://www.w3.org/TR/xml-exc-c14n/

mod attribute_axis;
mod engine;
mod namespace_axis;

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use oxixml_dom::Document;

use crate::error::C14nResult;
use crate::nodeset::NodeSet;

pub(crate) use engine::Engine;

/// Which of the three classic algorithms to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Variant {
    /// Canonical XML 1.0.
    Inclusive10,
    /// Canonical XML 1.1: `xml:base` fixup and no `xml:id` inheritance.
    Inclusive11,
    /// Exclusive XML Canonicalization 1.0.
    Exclusive(InclusiveNamespaces),
}

/// The `InclusiveNamespaces` `PrefixList` parameter of Exclusive C14N.
///
/// Prefixes on this list are "handled as provided in Canonical XML", i.e. they
/// are rendered whether or not the element visibly utilizes them. The special
/// token `#default` stands for the default namespace declaration.
///
/// # Example
///
/// ```
/// use oxixml_c14n::InclusiveNamespaces;
///
/// let list: InclusiveNamespaces = "wsu #default xsd".parse()?;
/// assert!(list.contains("wsu"));
/// assert!(list.contains_default());
/// assert!(!list.contains("wsse"));
/// // Round-trips through the canonical space-separated spelling, sorted.
/// assert_eq!(list.to_prefix_list(), "#default wsu xsd");
/// # Ok::<(), std::convert::Infallible>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InclusiveNamespaces {
    prefixes: BTreeSet<String>,
    default: bool,
}

impl InclusiveNamespaces {
    /// The token that stands for the default namespace declaration.
    pub const DEFAULT_TOKEN: &'static str = "#default";

    /// Returns the empty prefix list, which is what a missing
    /// `InclusiveNamespaces` element means.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Parses the whitespace-separated `PrefixList` attribute value.
    #[must_use]
    pub fn parse(prefix_list: &str) -> Self {
        prefix_list.split_whitespace().collect()
    }

    /// Adds one prefix, or the [`DEFAULT_TOKEN`](Self::DEFAULT_TOKEN).
    pub fn insert(&mut self, prefix: &str) -> &mut Self {
        if prefix == Self::DEFAULT_TOKEN || prefix.is_empty() {
            self.default = true;
        } else {
            self.prefixes.insert(prefix.to_owned());
        }
        self
    }

    /// Returns `true` when `prefix` is on the list; the empty string asks about
    /// the default namespace.
    #[must_use]
    pub fn contains(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            self.default
        } else {
            self.prefixes.contains(prefix)
        }
    }

    /// Returns `true` when the default namespace declaration is on the list.
    #[must_use]
    pub const fn contains_default(&self) -> bool {
        self.default
    }

    /// Returns `true` when the list names nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.default && self.prefixes.is_empty()
    }

    /// Iterates over the listed prefixes in ascending order, `#default` first.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.default
            .then_some(Self::DEFAULT_TOKEN)
            .into_iter()
            .chain(self.prefixes.iter().map(String::as_str))
    }

    /// Renders the list back into the `PrefixList` attribute syntax.
    #[must_use]
    pub fn to_prefix_list(&self) -> String {
        let mut out = String::new();
        for prefix in self.iter() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(prefix);
        }
        out
    }
}

impl FromStr for InclusiveNamespaces {
    type Err = std::convert::Infallible;

    fn from_str(prefix_list: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse(prefix_list))
    }
}

impl<S: AsRef<str>> FromIterator<S> for InclusiveNamespaces {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let mut list = Self::none();
        for prefix in iter {
            list.insert(prefix.as_ref());
        }
        list
    }
}

impl fmt::Display for InclusiveNamespaces {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_prefix_list())
    }
}

/// Runs one of the classic algorithms over a document subset.
pub(crate) fn canonicalize(
    document: &Document,
    subset: &NodeSet,
    variant: &Variant,
    with_comments: bool,
) -> C14nResult<Vec<u8>> {
    Engine::new(document, subset, variant, with_comments).run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_lists_round_trip() {
        let list = InclusiveNamespaces::parse("  b  #default a ");
        assert_eq!(list.to_prefix_list(), "#default a b");
        assert!(list.contains(""));
        assert!(!InclusiveNamespaces::none().contains(""));
        assert!(InclusiveNamespaces::none().is_empty());
    }
}
