//! [`ExpandedName`] / [`ExpandedNameRef`]: the resolved, namespace-URI-plus-local-name identity
//! of an XML name.

use std::fmt;
use std::hash::{Hash, Hasher};

/// A resolved XML name: an optional namespace name plus a local name, owning its strings.
///
/// This is the universal name identity used once a lexical [`crate::QName`] has been resolved
/// against a namespace context — the DOM, XPath, and schema layers all key on
/// `(namespace, local_name)`, never on which prefix happened to be used to spell the name in a
/// particular document. [`Eq`] and [`std::hash::Hash`] are defined on exactly that pair, so two
/// `ExpandedName`s compare equal whenever their namespace and local name match, regardless of
/// prefix.
///
/// An `ExpandedName` may additionally *retain* a lexical prefix, attached via
/// [`with_prefix`](Self::with_prefix) and read back with [`prefix`](Self::prefix). This exists so
/// that call sites which resolved a name from a prefixed `QName` — or that constructed one, as an
/// XQuery computed element/attribute constructor does — can recover the spelling that was
/// actually used, e.g. to implement `fn:name()` or `fn:prefix-from-QName()` per
/// [XDM 3.1 §5.4](https://www.w3.org/TR/xpath-datamodel-31/#QName). The prefix is a pure
/// annotation: it is invisible to [`PartialEq`], [`Eq`], and [`std::hash::Hash`], which — exactly
/// as before this annotation existed — still compare and hash only `(namespace, local_name)`.
///
/// See [`ExpandedNameRef`] for a borrowed, allocation-free view with the same semantics.
///
/// # Examples
///
/// ```
/// use oxixml_qname::ExpandedName;
///
/// let a = ExpandedName::new(Some("urn:example:book"), "title");
/// let b = ExpandedName::new(Some("urn:example:book"), "title");
/// let c = ExpandedName::new(None::<&str>, "title");
///
/// assert_eq!(a, b); // same namespace + local name: equal, regardless of how each was built
/// assert_ne!(a, c); // different namespace: unequal
/// assert_eq!(a.namespace(), Some("urn:example:book"));
/// assert_eq!(a.local_name(), "title");
/// ```
///
/// # Prefix retention is invisible to equality and hashing
///
/// ```
/// use oxixml_qname::ExpandedName;
/// use std::collections::HashSet;
/// use std::collections::hash_map::DefaultHasher;
/// use std::hash::{Hash, Hasher};
///
/// fn hash_of(name: &ExpandedName) -> u64 {
///     let mut hasher = DefaultHasher::new();
///     name.hash(&mut hasher);
///     hasher.finish()
/// }
///
/// // Three spellings of "the same" expanded name: two different retained prefixes, and none.
/// let via_a = ExpandedName::new(Some("urn:shared"), "x").with_prefix("a");
/// let via_b = ExpandedName::new(Some("urn:shared"), "x").with_prefix("b");
/// let bare = ExpandedName::new(Some("urn:shared"), "x");
///
/// // Equal-modulo-prefix, both ways...
/// assert_eq!(via_a, via_b);
/// assert_eq!(via_a, bare);
/// assert_eq!(via_b, bare);
/// // ...and equal hashes, so they are interchangeable as HashSet/HashMap keys.
/// assert_eq!(hash_of(&via_a), hash_of(&via_b));
/// assert_eq!(hash_of(&via_a), hash_of(&bare));
///
/// let mut set = HashSet::new();
/// set.insert(via_a.clone());
/// assert!(set.contains(&via_b));
/// assert!(set.contains(&bare));
/// assert_eq!(set.len(), 1);
///
/// // Yet each value still remembers its own prefix independently.
/// assert_eq!(via_a.prefix(), Some("a"));
/// assert_eq!(via_b.prefix(), Some("b"));
/// assert_eq!(bare.prefix(), None);
/// ```
#[derive(Debug, Clone)]
pub struct ExpandedName {
    namespace: Option<Box<str>>,
    local: Box<str>,
    prefix: Option<Box<str>>,
}

impl ExpandedName {
    /// Builds an `ExpandedName` from a namespace name and a local name.
    ///
    /// The result has no retained prefix — its [`prefix`](Self::prefix) is `None` until one is
    /// attached with [`with_prefix`](Self::with_prefix).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::ExpandedName;
    ///
    /// let no_namespace = ExpandedName::new(None::<&str>, "id");
    /// assert_eq!(no_namespace.namespace(), None);
    /// assert_eq!(no_namespace.prefix(), None);
    ///
    /// let namespaced = ExpandedName::new(Some("urn:example:book"), "title");
    /// assert_eq!(namespaced.namespace(), Some("urn:example:book"));
    /// ```
    pub fn new(namespace: Option<impl Into<String>>, local: impl Into<String>) -> Self {
        Self {
            namespace: namespace.map(|namespace| namespace.into().into_boxed_str()),
            local: local.into().into_boxed_str(),
            prefix: None,
        }
    }

    /// The namespace name, or `None` if this name is not in any namespace.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// The local name (the part after the namespace, corresponding to a `QName`'s local part).
    #[must_use]
    pub fn local_name(&self) -> &str {
        &self.local
    }

    /// Returns this name with a retained lexical prefix attached.
    ///
    /// The prefix is a pure annotation recording *one* spelling this name has been written with
    /// (e.g. the prefix of the `QName` it was resolved from, or the one a computed constructor
    /// asked for). It does not participate in [`PartialEq`], [`Eq`], or [`std::hash::Hash`] — see
    /// the type-level documentation of [`ExpandedName`] for a doctest proving this — so attaching
    /// or omitting it never changes whether two `ExpandedName`s are equal, or which bucket of a
    /// `HashSet`/`HashMap` one falls into.
    ///
    /// Calling this again replaces whatever prefix was previously attached, rather than
    /// accumulating; there is only ever at most one retained prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::ExpandedName;
    ///
    /// let name = ExpandedName::new(Some("urn:example:book"), "title").with_prefix("book");
    /// assert_eq!(name.prefix(), Some("book"));
    /// assert_eq!(name.namespace(), Some("urn:example:book")); // unaffected
    ///
    /// // A later call replaces, rather than adds to, the retained prefix.
    /// let reprefixed = name.with_prefix("b");
    /// assert_eq!(reprefixed.prefix(), Some("b"));
    /// ```
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into().into_boxed_str());
        self
    }

    /// The lexical prefix retained via [`with_prefix`](Self::with_prefix), or `None` if none was
    /// ever attached.
    ///
    /// This is purely informational: it plays no part in this type's [`PartialEq`], [`Eq`], or
    /// [`std::hash::Hash`] implementations (see the type-level documentation of [`ExpandedName`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::ExpandedName;
    ///
    /// assert_eq!(ExpandedName::new(None::<&str>, "id").prefix(), None);
    /// assert_eq!(
    ///     ExpandedName::new(None::<&str>, "id").with_prefix("p").prefix(),
    ///     Some("p")
    /// );
    /// ```
    #[must_use]
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// Returns a borrowed [`ExpandedNameRef`] view of this name.
    ///
    /// The retained [`prefix`](Self::prefix), if any, is carried over into the returned view.
    #[must_use]
    pub fn as_ref(&self) -> ExpandedNameRef<'_> {
        ExpandedNameRef {
            namespace: self.namespace.as_deref(),
            local: &self.local,
            prefix: self.prefix.as_deref(),
        }
    }
}

impl PartialEq for ExpandedName {
    /// Compares by `(namespace, local_name)` only — the retained [`prefix`](Self::prefix), if
    /// any, is deliberately excluded; see the type-level documentation of [`ExpandedName`].
    fn eq(&self, other: &Self) -> bool {
        self.namespace == other.namespace && self.local == other.local
    }
}

impl Eq for ExpandedName {}

impl Hash for ExpandedName {
    /// Hashes `(namespace, local_name)` only, so that this stays consistent with [`Eq`] — the
    /// retained [`prefix`](Self::prefix), if any, is deliberately excluded; see the type-level
    /// documentation of [`ExpandedName`].
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.local.hash(state);
    }
}

impl fmt::Display for ExpandedName {
    /// Formats using [James Clark's notation](http://www.jclark.com/xml/xmlns.htm),
    /// `{namespace}local`, or just `local` when there is no namespace.
    ///
    /// The retained [`prefix`](Self::prefix), if any, never appears here — Clark notation has no
    /// place for it, by design (that is exactly the point of resolving to a prefix-independent
    /// identity in the first place).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<'a> PartialEq<ExpandedNameRef<'a>> for ExpandedName {
    fn eq(&self, other: &ExpandedNameRef<'a>) -> bool {
        self.namespace() == other.namespace && &*self.local == other.local
    }
}

impl<'a> From<ExpandedNameRef<'a>> for ExpandedName {
    fn from(reference: ExpandedNameRef<'a>) -> Self {
        reference.into_owned()
    }
}

/// A borrowed, allocation-free view of an [`ExpandedName`].
///
/// Both fields share the single lifetime `'a`: build one by hand with [`ExpandedNameRef::new`],
/// via [`ExpandedName::as_ref`], or from [`crate::NamespaceStack::resolve_qname_ref`], which
/// borrows the namespace name from the stack itself and the local name from the input `QName`
/// text — see that method's documentation for the lifetime consequences of sharing `'a` between
/// two otherwise-independent borrows.
///
/// Like [`ExpandedName`], this type can retain a lexical prefix via
/// [`with_prefix`](Self::with_prefix); the prefix plays no part in [`PartialEq`], [`Eq`], or
/// [`std::hash::Hash`], which compare and hash `(namespace, local_name)` only, exactly as for the
/// owned type.
///
/// # Examples
///
/// ```
/// use oxixml_qname::ExpandedNameRef;
///
/// let a = ExpandedNameRef::new(Some("urn:example:book"), "title");
/// let b = ExpandedNameRef::new(Some("urn:example:book"), "title");
/// assert_eq!(a, b);
/// assert_eq!(a.namespace(), Some("urn:example:book"));
/// assert_eq!(a.local_name(), "title");
///
/// let owned = a.into_owned();
/// assert_eq!(owned, a);
/// ```
///
/// # Prefix retention is invisible to equality and hashing
///
/// ```
/// use oxixml_qname::ExpandedNameRef;
/// use std::collections::hash_map::DefaultHasher;
/// use std::hash::{Hash, Hasher};
///
/// fn hash_of(name: ExpandedNameRef<'_>) -> u64 {
///     let mut hasher = DefaultHasher::new();
///     name.hash(&mut hasher);
///     hasher.finish()
/// }
///
/// let via_a = ExpandedNameRef::new(Some("urn:shared"), "x").with_prefix("a");
/// let via_b = ExpandedNameRef::new(Some("urn:shared"), "x").with_prefix("b");
/// let bare = ExpandedNameRef::new(Some("urn:shared"), "x");
///
/// assert_eq!(via_a, via_b);
/// assert_eq!(via_a, bare);
/// assert_eq!(hash_of(via_a), hash_of(via_b));
/// assert_eq!(hash_of(via_a), hash_of(bare));
///
/// assert_eq!(via_a.prefix(), Some("a"));
/// assert_eq!(via_b.prefix(), Some("b"));
/// assert_eq!(bare.prefix(), None);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExpandedNameRef<'a> {
    namespace: Option<&'a str>,
    local: &'a str,
    prefix: Option<&'a str>,
}

impl<'a> ExpandedNameRef<'a> {
    /// Builds an `ExpandedNameRef` from a borrowed namespace name and local name.
    ///
    /// The result has no retained prefix — its [`prefix`](Self::prefix) is `None` until one is
    /// attached with [`with_prefix`](Self::with_prefix).
    #[must_use]
    pub fn new(namespace: Option<&'a str>, local: &'a str) -> Self {
        Self {
            namespace,
            local,
            prefix: None,
        }
    }

    /// The namespace name, or `None` if this name is not in any namespace.
    #[must_use]
    pub fn namespace(&self) -> Option<&'a str> {
        self.namespace
    }

    /// The local name.
    #[must_use]
    pub fn local_name(&self) -> &'a str {
        self.local
    }

    /// Returns this borrowed name with a retained lexical prefix attached.
    ///
    /// See [`ExpandedName::with_prefix`] for the full contract: this is the same annotation,
    /// borrowed rather than owned, and just as invisible to [`PartialEq`], [`Eq`], and
    /// [`std::hash::Hash`]. Calling this again replaces whatever prefix was previously attached.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_qname::ExpandedNameRef;
    ///
    /// let name = ExpandedNameRef::new(Some("urn:example:book"), "title").with_prefix("book");
    /// assert_eq!(name.prefix(), Some("book"));
    /// assert_eq!(name.namespace(), Some("urn:example:book")); // unaffected
    /// ```
    #[must_use]
    pub fn with_prefix(mut self, prefix: &'a str) -> Self {
        self.prefix = Some(prefix);
        self
    }

    /// The lexical prefix retained via [`with_prefix`](Self::with_prefix), or `None` if none was
    /// ever attached.
    ///
    /// This is purely informational: it plays no part in this type's [`PartialEq`], [`Eq`], or
    /// [`std::hash::Hash`] implementations.
    #[must_use]
    pub fn prefix(&self) -> Option<&'a str> {
        self.prefix
    }

    /// Allocates an owned [`ExpandedName`] with the same namespace and local name, carrying over
    /// the retained [`prefix`](Self::prefix) as well, if any.
    ///
    /// `ExpandedNameRef` is [`Copy`], so this takes `self` by value rather than shadowing
    /// [`ToOwned::to_owned`] (which takes `&self` and, via the standard blanket
    /// `impl<T: Clone> ToOwned for T`, would otherwise just hand back another
    /// `ExpandedNameRef`).
    #[must_use]
    pub fn into_owned(self) -> ExpandedName {
        ExpandedName {
            namespace: self.namespace.map(Box::from),
            local: Box::from(self.local),
            prefix: self.prefix.map(Box::from),
        }
    }
}

impl PartialEq for ExpandedNameRef<'_> {
    /// Compares by `(namespace, local_name)` only — the retained [`prefix`](Self::prefix), if
    /// any, is deliberately excluded; see the type-level documentation of [`ExpandedName`].
    fn eq(&self, other: &Self) -> bool {
        self.namespace == other.namespace && self.local == other.local
    }
}

impl Eq for ExpandedNameRef<'_> {}

impl Hash for ExpandedNameRef<'_> {
    /// Hashes `(namespace, local_name)` only, so that this stays consistent with [`Eq`] — the
    /// retained [`prefix`](Self::prefix), if any, is deliberately excluded.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.local.hash(state);
    }
}

impl fmt::Display for ExpandedNameRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.namespace {
            Some(namespace) => write!(f, "{{{namespace}}}{}", self.local),
            None => write!(f, "{}", self.local),
        }
    }
}

impl<'a> PartialEq<ExpandedName> for ExpandedNameRef<'a> {
    fn eq(&self, other: &ExpandedName) -> bool {
        self.namespace == other.namespace() && self.local == &*other.local
    }
}

impl<'a> From<&'a ExpandedName> for ExpandedNameRef<'a> {
    fn from(name: &'a ExpandedName) -> Self {
        name.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::collections::hash_map::DefaultHasher;

    #[test]
    fn equality_ignores_how_it_was_built() {
        let a = ExpandedName::new(Some("urn:ns"), "local");
        let b = ExpandedName::new(Some("urn:ns".to_owned()), "local".to_owned());
        assert_eq!(a, b);
    }

    #[test]
    fn equality_is_sensitive_to_namespace_and_local() {
        let base = ExpandedName::new(Some("urn:ns"), "local");
        assert_ne!(base, ExpandedName::new(Some("urn:other"), "local"));
        assert_ne!(base, ExpandedName::new(None::<&str>, "local"));
        assert_ne!(base, ExpandedName::new(Some("urn:ns"), "other"));
    }

    #[test]
    fn hash_agrees_with_eq_across_different_prefixes_producing_the_same_identity() {
        // Simulate two QNames with different prefixes (`a:x` and `b:x`) that both resolved to
        // the same namespace: the resulting ExpandedNames must be the same set member.
        let via_prefix_a = ExpandedName::new(Some("urn:shared"), "x");
        let via_prefix_b = ExpandedName::new(Some("urn:shared"), "x");

        let mut set = HashSet::new();
        set.insert(via_prefix_a);
        assert!(set.contains(&via_prefix_b));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn ref_and_owned_compare_equal_both_ways() {
        let owned = ExpandedName::new(Some("urn:ns"), "local");
        let borrowed = ExpandedNameRef::new(Some("urn:ns"), "local");
        assert_eq!(owned, borrowed);
        assert_eq!(borrowed, owned);
    }

    #[test]
    fn as_ref_and_to_owned_roundtrip() {
        let owned = ExpandedName::new(Some("urn:ns"), "local");
        let borrowed = owned.as_ref();
        assert_eq!(borrowed.namespace(), Some("urn:ns"));
        assert_eq!(borrowed.local_name(), "local");
        assert_eq!(borrowed.into_owned(), owned);
    }

    #[test]
    fn display_uses_clark_notation() {
        let namespaced = ExpandedName::new(Some("urn:example:book"), "title");
        assert_eq!(namespaced.to_string(), "{urn:example:book}title");

        let unnamespaced = ExpandedName::new(None::<&str>, "title");
        assert_eq!(unnamespaced.to_string(), "title");
    }

    #[test]
    fn from_conversions() {
        let owned = ExpandedName::new(Some("urn:ns"), "local");
        let reference: ExpandedNameRef<'_> = (&owned).into();
        let round_tripped: ExpandedName = reference.into();
        assert_eq!(owned, round_tripped);
    }

    // --- Prefix retention -------------------------------------------------------------------
    //
    // `ExpandedName`/`ExpandedNameRef` have never implemented `Ord`/`PartialOrd` (only
    // `PartialEq`/`Eq`/`Hash`, exercised above and below) — there is no workspace usage that
    // sorts or `BTreeMap`/`BTreeSet`-keys either type. Per the additive-only policy for this
    // change, no `Ord`/`PartialOrd` impl is introduced here either, so there is no ordering
    // behavior for the retained prefix to disturb.

    #[test]
    fn prefix_defaults_to_none() {
        assert_eq!(ExpandedName::new(Some("urn:ns"), "local").prefix(), None);
        assert_eq!(ExpandedNameRef::new(Some("urn:ns"), "local").prefix(), None);
    }

    #[test]
    fn with_prefix_round_trip() {
        let name = ExpandedName::new(Some("urn:ns"), "local").with_prefix("p");
        assert_eq!(name.prefix(), Some("p"));
        assert_eq!(name.namespace(), Some("urn:ns")); // unaffected
        assert_eq!(name.local_name(), "local"); // unaffected

        let borrowed = ExpandedNameRef::new(Some("urn:ns"), "local").with_prefix("p");
        assert_eq!(borrowed.prefix(), Some("p"));
        assert_eq!(borrowed.namespace(), Some("urn:ns"));
        assert_eq!(borrowed.local_name(), "local");
    }

    #[test]
    fn with_prefix_replaces_rather_than_accumulates() {
        let name = ExpandedName::new(None::<&str>, "local")
            .with_prefix("a")
            .with_prefix("b");
        assert_eq!(name.prefix(), Some("b"));

        let borrowed = ExpandedNameRef::new(None, "local")
            .with_prefix("a")
            .with_prefix("b");
        assert_eq!(borrowed.prefix(), Some("b"));
    }

    #[test]
    fn prefix_does_not_affect_equality() {
        let unprefixed = ExpandedName::new(Some("urn:ns"), "local");
        let prefixed_a = unprefixed.clone().with_prefix("a");
        let prefixed_b = unprefixed.clone().with_prefix("b");

        assert_eq!(unprefixed, prefixed_a);
        assert_eq!(prefixed_a, prefixed_b);
        assert_eq!(prefixed_b, unprefixed);

        // Distinct retained prefixes remain independently observable even while equal overall.
        assert_ne!(prefixed_a.prefix(), prefixed_b.prefix());
    }

    #[test]
    fn prefix_ref_does_not_affect_equality() {
        let unprefixed = ExpandedNameRef::new(Some("urn:ns"), "local");
        let prefixed_a = unprefixed.with_prefix("a");
        let prefixed_b = unprefixed.with_prefix("b");

        assert_eq!(unprefixed, prefixed_a);
        assert_eq!(prefixed_a, prefixed_b);
        assert_ne!(prefixed_a.prefix(), prefixed_b.prefix());
    }

    #[test]
    fn prefix_does_not_affect_hash() {
        fn hash_of(name: &ExpandedName) -> u64 {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        }

        let unprefixed = ExpandedName::new(Some("urn:ns"), "local");
        let prefixed_a = unprefixed.clone().with_prefix("a");
        let prefixed_b = unprefixed.clone().with_prefix("b");

        assert_eq!(hash_of(&unprefixed), hash_of(&prefixed_a));
        assert_eq!(hash_of(&unprefixed), hash_of(&prefixed_b));
        assert_eq!(hash_of(&prefixed_a), hash_of(&prefixed_b));
    }

    #[test]
    fn prefix_ref_does_not_affect_hash() {
        fn hash_of(name: ExpandedNameRef<'_>) -> u64 {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        }

        let unprefixed = ExpandedNameRef::new(Some("urn:ns"), "local");
        let prefixed_a = unprefixed.with_prefix("a");
        let prefixed_b = unprefixed.with_prefix("b");

        assert_eq!(hash_of(unprefixed), hash_of(prefixed_a));
        assert_eq!(hash_of(prefixed_a), hash_of(prefixed_b));
    }

    #[test]
    fn prefix_ignored_by_hashset_membership() {
        let mut set = HashSet::new();
        set.insert(ExpandedName::new(Some("urn:shared"), "x").with_prefix("a"));

        // A lookup with a *different* retained prefix (or none at all) still finds it: the
        // prefix is not part of the set's notion of identity.
        assert!(set.contains(&ExpandedName::new(Some("urn:shared"), "x").with_prefix("b")));
        assert!(set.contains(&ExpandedName::new(Some("urn:shared"), "x")));
        assert_eq!(set.len(), 1);

        // Inserting an equal-modulo-prefix value again does not grow the set.
        let newly_inserted =
            set.insert(ExpandedName::new(Some("urn:shared"), "x").with_prefix("c"));
        assert!(!newly_inserted);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn prefix_ignored_by_hashmap_key_lookup() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(
            ExpandedName::new(Some("urn:shared"), "x").with_prefix("a"),
            1,
        );

        assert_eq!(
            map.get(&ExpandedName::new(Some("urn:shared"), "x").with_prefix("different")),
            Some(&1)
        );
        assert_eq!(
            map.get(&ExpandedName::new(Some("urn:shared"), "x")),
            Some(&1)
        );
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn as_ref_and_into_owned_carry_the_prefix_through() {
        let owned = ExpandedName::new(Some("urn:ns"), "local").with_prefix("p");
        let borrowed = owned.as_ref();
        assert_eq!(borrowed.prefix(), Some("p"));

        let round_tripped = borrowed.into_owned();
        assert_eq!(round_tripped.prefix(), Some("p"));
        assert_eq!(round_tripped, owned); // equal regardless, but confirms no drift either
    }

    #[test]
    fn cross_type_equality_ignores_prefix_in_both_directions() {
        let owned = ExpandedName::new(Some("urn:ns"), "local").with_prefix("owned-prefix");
        let borrowed = ExpandedNameRef::new(Some("urn:ns"), "local").with_prefix("ref-prefix");

        assert_eq!(owned, borrowed);
        assert_eq!(borrowed, owned);
    }

    #[test]
    fn display_ignores_retained_prefix() {
        let namespaced = ExpandedName::new(Some("urn:example:book"), "title").with_prefix("book");
        assert_eq!(namespaced.to_string(), "{urn:example:book}title");

        let unnamespaced = ExpandedName::new(None::<&str>, "title").with_prefix("x");
        assert_eq!(unnamespaced.to_string(), "title");
    }
}
