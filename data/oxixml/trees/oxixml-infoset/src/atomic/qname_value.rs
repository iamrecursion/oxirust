//! [`QNameValue`] — the value space of `xs:QName` and `xs:NOTATION`.

use oxixml_qname::{ExpandedName, ExpandedNameRef};
use std::fmt;
use std::hash::{Hash, Hasher};

/// The value of an `xs:QName` (or `xs:NOTATION`) atomic value.
///
/// Per [XML Schema 1.1 Part 2 §3.3.18](https://www.w3.org/TR/xmlschema11-2/#QName), the *value*
/// of an `xs:QName` is the pair `(namespace name, local name)`; the prefix is retained only so
/// that the value can be serialized back to a lexical form, and it takes no part in equality.
/// XDM 3.1 §3.3.2 says the same thing: "the prefix … is not part of the value; it is retained
/// for use in serialization".
///
/// # Examples
///
/// ```
/// use oxixml_infoset::QNameValue;
///
/// let with_prefix = QNameValue::new(Some("bk"), Some("urn:example:book"), "title");
/// let without_prefix = QNameValue::new(None::<&str>, Some("urn:example:book"), "title");
///
/// // The prefix does not participate in equality...
/// assert_eq!(with_prefix, without_prefix);
/// // ...but it does determine the lexical form.
/// assert_eq!(with_prefix.lexical_form(), "bk:title");
/// assert_eq!(without_prefix.lexical_form(), "title");
/// ```
#[derive(Debug, Clone, Eq)]
pub struct QNameValue {
    prefix: Option<String>,
    namespace: Option<String>,
    local: String,
}

impl QNameValue {
    /// Builds a `QNameValue` from an optional prefix, an optional namespace name, and a local
    /// name.
    ///
    /// A `Some` prefix with a `None` namespace is accepted (it can arise while a document is
    /// still being resolved) but is never produced by this crate's own tree construction.
    pub fn new(
        prefix: Option<impl Into<String>>,
        namespace: Option<impl Into<String>>,
        local: impl Into<String>,
    ) -> Self {
        Self {
            prefix: prefix.map(Into::into),
            namespace: namespace.map(Into::into),
            local: local.into(),
        }
    }

    /// The prefix used to spell this name, if any. Not part of the value.
    #[must_use]
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// The namespace name, or `None` if the name is in no namespace.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// The local name.
    #[must_use]
    pub fn local_name(&self) -> &str {
        &self.local
    }

    /// The `(namespace, local)` pair that *is* the value, as an [`ExpandedName`].
    #[must_use]
    pub fn expanded_name(&self) -> ExpandedName {
        ExpandedName::new(self.namespace.clone(), self.local.clone())
    }

    /// A borrowed view of the `(namespace, local)` pair.
    #[must_use]
    pub fn expanded_name_ref(&self) -> ExpandedNameRef<'_> {
        ExpandedNameRef::new(self.namespace.as_deref(), &self.local)
    }

    /// The lexical form: `prefix:local` when a prefix is present, `local` otherwise.
    ///
    /// This is the string value of an `xs:QName` atomic value.
    #[must_use]
    pub fn lexical_form(&self) -> String {
        match &self.prefix {
            Some(prefix) => format!("{prefix}:{}", self.local),
            None => self.local.clone(),
        }
    }
}

impl PartialEq for QNameValue {
    /// Compares the `(namespace, local)` pair only; the prefix is ignored.
    fn eq(&self, other: &Self) -> bool {
        self.namespace == other.namespace && self.local == other.local
    }
}

impl Hash for QNameValue {
    /// Hashes the `(namespace, local)` pair only, to stay consistent with [`PartialEq`].
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.local.hash(state);
    }
}

impl fmt::Display for QNameValue {
    /// Formats the lexical form (see [`QNameValue::lexical_form`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.prefix {
            Some(prefix) => write!(f, "{prefix}:{}", self.local),
            None => f.write_str(&self.local),
        }
    }
}

impl From<ExpandedName> for QNameValue {
    /// An [`ExpandedName`] may *retain* the prefix it was resolved from, and XDM 3.1 §5.4 makes
    /// that prefix the third component of an `xs:QName`, so it is carried across rather than
    /// dropped: `fn:node-name` on an element written `pre:e` answers `pre:e`.
    fn from(name: ExpandedName) -> Self {
        Self {
            prefix: name.prefix().map(str::to_owned),
            namespace: name.namespace().map(str::to_owned),
            local: name.local_name().to_owned(),
        }
    }
}

impl From<QNameValue> for ExpandedName {
    fn from(value: QNameValue) -> Self {
        Self::new(value.namespace, value.local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(value: &QNameValue) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn prefix_is_ignored_by_equality_and_hashing() {
        let a = QNameValue::new(Some("p"), Some("urn:ns"), "local");
        let b = QNameValue::new(Some("q"), Some("urn:ns"), "local");
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn namespace_and_local_participate_in_equality() {
        let base = QNameValue::new(None::<&str>, Some("urn:ns"), "local");
        assert_ne!(
            base,
            QNameValue::new(None::<&str>, Some("urn:other"), "local")
        );
        assert_ne!(base, QNameValue::new(None::<&str>, None::<&str>, "local"));
        assert_ne!(base, QNameValue::new(None::<&str>, Some("urn:ns"), "other"));
    }

    #[test]
    fn lexical_form_and_display_agree() {
        let value = QNameValue::new(Some("bk"), Some("urn:example:book"), "title");
        assert_eq!(value.lexical_form(), "bk:title");
        assert_eq!(value.to_string(), "bk:title");
        assert_eq!(value.prefix(), Some("bk"));
        assert_eq!(value.namespace(), Some("urn:example:book"));
        assert_eq!(value.local_name(), "title");
    }

    #[test]
    fn expanded_name_conversions_round_trip_the_value() {
        let value = QNameValue::new(Some("bk"), Some("urn:example:book"), "title");
        let expanded = value.expanded_name();
        assert_eq!(expanded.namespace(), Some("urn:example:book"));
        assert_eq!(expanded.local_name(), "title");
        assert_eq!(value.expanded_name_ref(), expanded.as_ref());

        let back: QNameValue = expanded.into();
        assert_eq!(back, value);
        assert_eq!(back.prefix(), None, "the prefix is not recoverable");
    }
}
