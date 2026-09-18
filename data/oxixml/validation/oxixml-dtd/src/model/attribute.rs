//! Attribute-list declarations.
//!
//! Mirrors the `AttlistDecl` production of XML 1.0 §3.3 `[52]` and its
//! `AttType` `[54]` and `DefaultDecl` `[60]` sub-productions.

use core::fmt;
use std::collections::HashSet;

/// The declared type of an attribute (XML 1.0 §3.3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributeType {
    /// `CDATA` — any string.
    Cdata,
    /// `ID` — a `Name`, unique across the document.
    Id,
    /// `IDREF` — a `Name` matching some `ID` in the document.
    IdRef,
    /// `IDREFS` — whitespace-separated `IDREF`s.
    IdRefs,
    /// `ENTITY` — the name of a declared unparsed entity.
    Entity,
    /// `ENTITIES` — whitespace-separated `ENTITY` names.
    Entities,
    /// `NMTOKEN` — an `Nmtoken`.
    NmToken,
    /// `NMTOKENS` — whitespace-separated `Nmtoken`s.
    NmTokens,
    /// `NOTATION (a | b)` — one of the listed, declared notation names.
    Notation(Vec<String>),
    /// `(a | b)` — one of the listed `Nmtoken`s.
    Enumeration(Vec<String>),
}

impl AttributeType {
    /// Returns `true` for the types whose values are subject to the
    /// "further normalization" of XML 1.0 §3.3.3 — every type except `CDATA`.
    ///
    /// For those types, leading and trailing spaces are discarded and internal
    /// runs of spaces collapse to a single space.
    #[must_use]
    pub const fn is_tokenized(&self) -> bool {
        !matches!(self, Self::Cdata)
    }

    /// Returns `true` when a value of this type is a whitespace-separated list.
    #[must_use]
    pub const fn is_list(&self) -> bool {
        matches!(self, Self::IdRefs | Self::Entities | Self::NmTokens)
    }

    /// The tokens of a `NOTATION` or enumeration type, otherwise `None`.
    #[must_use]
    pub fn tokens(&self) -> Option<&[String]> {
        match self {
            Self::Notation(tokens) | Self::Enumeration(tokens) => Some(tokens),
            _ => None,
        }
    }

    /// The keyword that spells this type, without any token list.
    #[must_use]
    pub const fn keyword(&self) -> &'static str {
        match self {
            Self::Cdata => "CDATA",
            Self::Id => "ID",
            Self::IdRef => "IDREF",
            Self::IdRefs => "IDREFS",
            Self::Entity => "ENTITY",
            Self::Entities => "ENTITIES",
            Self::NmToken => "NMTOKEN",
            Self::NmTokens => "NMTOKENS",
            Self::Notation(_) => "NOTATION",
            Self::Enumeration(_) => "",
        }
    }

    /// Returns the first token listed twice, if any — a violation of
    /// VC: No Duplicate Tokens (XML 1.0 §3.3.1).
    #[must_use]
    pub fn first_duplicate_token(&self) -> Option<&str> {
        let tokens = self.tokens()?;
        // A `HashSet` built in one pass keeps this linear in the token count; the nested scan
        // it replaces was quadratic, which mattered once enumeration/notation lists ran into
        // the thousands of tokens.
        let mut seen: HashSet<&str> = HashSet::with_capacity(tokens.len());
        tokens
            .iter()
            .find(|token| !seen.insert(token.as_str()))
            .map(String::as_str)
    }
}

impl fmt::Display for AttributeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Notation(tokens) => {
                f.write_str("NOTATION (")?;
                write_tokens(f, tokens)
            }
            Self::Enumeration(tokens) => {
                f.write_str("(")?;
                write_tokens(f, tokens)
            }
            other => f.write_str(other.keyword()),
        }
    }
}

/// Writes `a | b | c)`.
fn write_tokens(f: &mut fmt::Formatter<'_>, tokens: &[String]) -> fmt::Result {
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            f.write_str(" | ")?;
        }
        f.write_str(token)?;
    }
    f.write_str(")")
}

/// The default declaration of an attribute (XML 1.0 §3.3.2 `[60]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultDecl {
    /// `#REQUIRED` — every occurrence of the element must supply the attribute.
    Required,
    /// `#IMPLIED` — no default; the attribute may be absent.
    Implied,
    /// `#FIXED "value"` — the attribute, if present, must have exactly this
    /// (already normalized) value; if absent, it defaults to it.
    Fixed(String),
    /// `"value"` — the (already normalized) value used when the attribute is
    /// absent.
    Default(String),
}

impl DefaultDecl {
    /// The default value supplied when the attribute is absent, if any.
    #[must_use]
    pub fn value(&self) -> Option<&str> {
        match self {
            Self::Fixed(value) | Self::Default(value) => Some(value),
            Self::Required | Self::Implied => None,
        }
    }

    /// Returns `true` for `#REQUIRED`.
    #[must_use]
    pub const fn is_required(&self) -> bool {
        matches!(self, Self::Required)
    }

    /// Returns `true` for `#FIXED`.
    #[must_use]
    pub const fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}

impl fmt::Display for DefaultDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Required => f.write_str("#REQUIRED"),
            Self::Implied => f.write_str("#IMPLIED"),
            Self::Fixed(value) => write!(f, "#FIXED \"{value}\""),
            Self::Default(value) => write!(f, "\"{value}\""),
        }
    }
}

/// One `AttDef` of an attribute-list declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeDecl {
    name: String,
    att_type: AttributeType,
    default: DefaultDecl,
    external: bool,
}

impl AttributeDecl {
    /// Builds an attribute definition. `external` records whether it came from
    /// the external subset or a parameter entity (XML 1.0 §2.9).
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        att_type: AttributeType,
        default: DefaultDecl,
        external: bool,
    ) -> Self {
        Self {
            name: name.into(),
            att_type,
            default,
            external,
        }
    }

    /// The attribute name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The declared type.
    #[must_use]
    pub fn att_type(&self) -> &AttributeType {
        &self.att_type
    }

    /// The default declaration.
    #[must_use]
    pub fn default(&self) -> &DefaultDecl {
        &self.default
    }

    /// Whether the definition came from outside the internal subset.
    #[must_use]
    pub fn is_external(&self) -> bool {
        self.external
    }

    /// Replaces the default declaration.
    ///
    /// Used by the parser, which stores the *unnormalized* literal first and
    /// rewrites it once the whole DTD is known — attribute-value normalization
    /// may reference general entities that are declared further down.
    pub(crate) fn set_default(&mut self, default: DefaultDecl) {
        self.default = default;
    }
}

impl fmt::Display for AttributeDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.name, self.att_type, self.default)
    }
}

/// Every attribute declared for one element type.
///
/// XML 1.0 §3.3 allows an element type to have several `<!ATTLIST>`
/// declarations; they are merged here, and — per the same section — when the
/// same attribute is defined more than once, **the first definition wins** and
/// the later ones are discarded (a processor may warn, but must not error).
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{DefaultDecl, DtdParser};
///
/// let dtd = DtdParser::new().parse_internal_subset(
///     r#"<!ATTLIST a x CDATA "1">
///        <!ATTLIST a x CDATA "2" y NMTOKEN #IMPLIED>"#,
/// )?;
/// let attributes = dtd.attributes("a").expect("declared");
/// // The first definition of `x` wins.
/// assert_eq!(
///     attributes.get("x").expect("declared").default(),
///     &DefaultDecl::Default("1".to_owned()),
/// );
/// assert!(attributes.get("y").is_some());
/// # Ok::<(), oxixml_dtd::DtdError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AttributeList {
    element: String,
    attributes: Vec<AttributeDecl>,
    /// The name of every attribute already declared for this element, kept in step with
    /// `attributes` by [`insert`](Self::insert) — the list's only mutator — so that a
    /// duplicate name is rejected in O(1) instead of scanning every prior declaration. An
    /// element type with many `<!ATTLIST>` declarations would otherwise make DTD parsing
    /// quadratic in the number of attributes declared for it.
    names: HashSet<String>,
}

impl AttributeList {
    /// Creates an empty list for `element`.
    #[must_use]
    pub fn new(element: impl Into<String>) -> Self {
        Self {
            element: element.into(),
            attributes: Vec::new(),
            names: HashSet::new(),
        }
    }

    /// The element type these attributes belong to.
    #[must_use]
    pub fn element(&self) -> &str {
        &self.element
    }

    /// Adds `declaration` unless an attribute of the same name is already
    /// defined, per the first-declaration-wins rule of XML 1.0 §3.3.
    ///
    /// Returns `true` when the declaration was kept.
    pub fn insert(&mut self, declaration: AttributeDecl) -> bool {
        if !self.names.insert(declaration.name().to_owned()) {
            return false;
        }
        self.attributes.push(declaration);
        true
    }

    /// Looks an attribute definition up by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&AttributeDecl> {
        self.attributes
            .iter()
            .find(|declaration| declaration.name() == name)
    }

    /// Mutable access to an attribute definition, by name.
    pub(crate) fn get_mut(&mut self, name: &str) -> Option<&mut AttributeDecl> {
        self.attributes
            .iter_mut()
            .find(|declaration| declaration.name == name)
    }

    /// Every definition, in declaration order.
    #[must_use]
    pub fn all(&self) -> &[AttributeDecl] {
        &self.attributes
    }

    /// Iterates over the definitions.
    pub fn iter(&self) -> core::slice::Iter<'_, AttributeDecl> {
        self.attributes.iter()
    }

    /// The number of defined attributes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.attributes.len()
    }

    /// Returns `true` when no attributes are defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// The attribute declared with type `ID`, if any.
    #[must_use]
    pub fn id_attribute(&self) -> Option<&AttributeDecl> {
        self.attributes
            .iter()
            .find(|declaration| matches!(declaration.att_type(), AttributeType::Id))
    }

    /// Every definition that supplies a default value.
    pub fn defaults(&self) -> impl Iterator<Item = &AttributeDecl> {
        self.attributes
            .iter()
            .filter(|declaration| declaration.default().value().is_some())
    }
}

impl<'a> IntoIterator for &'a AttributeList {
    type Item = &'a AttributeDecl;
    type IntoIter = core::slice::Iter<'a, AttributeDecl>;

    fn into_iter(self) -> Self::IntoIter {
        self.attributes.iter()
    }
}

impl fmt::Display for AttributeList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<!ATTLIST {}", self.element)?;
        for declaration in &self.attributes {
            write!(f, " {declaration}")?;
        }
        f.write_str(">")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_huge_enumeration_with_no_duplicate_checks_quickly_not_quadratically() {
        // The nested linear scan `first_duplicate_token` used to do was quadratic in the
        // token count; a `HashSet` built in one pass keeps it linear. 200 000 tokens (an
        // enumeration attribute type no real DTD comes close to) would take on the order of
        // half a minute the old way and a small fraction of a second this way, so a generous
        // bound cleanly tells the two apart without being sensitive to machine speed.
        let tokens: Vec<String> = (0..200_000).map(|i| format!("v{i}")).collect();
        let att_type = AttributeType::Enumeration(tokens);
        let started = std::time::Instant::now();
        let duplicate = att_type.first_duplicate_token();
        let elapsed = started.elapsed();
        assert_eq!(duplicate, None);
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "first_duplicate_token took {elapsed:?} for 200 000 distinct tokens"
        );
    }

    #[test]
    fn many_attribute_declarations_on_one_element_are_inserted_quickly_not_quadratically() {
        // `AttributeList::insert`'s duplicate check had the same quadratic shape as
        // `first_duplicate_token`, scanning every already-inserted attribute on each call.
        let mut list = AttributeList::new("a");
        let started = std::time::Instant::now();
        for i in 0..200_000 {
            list.insert(AttributeDecl::new(
                format!("a{i}"),
                AttributeType::Cdata,
                DefaultDecl::Implied,
                false,
            ));
        }
        let elapsed = started.elapsed();
        assert_eq!(list.len(), 200_000);
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "inserting 200 000 distinct attributes took {elapsed:?}"
        );
    }

    #[test]
    fn first_definition_of_an_attribute_wins() {
        let mut list = AttributeList::new("a");
        assert!(list.insert(AttributeDecl::new(
            "x",
            AttributeType::Cdata,
            DefaultDecl::Default("1".to_owned()),
            false,
        )));
        assert!(!list.insert(AttributeDecl::new(
            "x",
            AttributeType::NmToken,
            DefaultDecl::Implied,
            false,
        )));
        assert_eq!(list.len(), 1);
        assert_eq!(
            list.get("x").map(AttributeDecl::att_type),
            Some(&AttributeType::Cdata)
        );
    }

    #[test]
    fn enumeration_duplicate_tokens_are_reported() {
        let att_type =
            AttributeType::Enumeration(vec!["a".to_owned(), "b".to_owned(), "a".to_owned()]);
        assert_eq!(att_type.first_duplicate_token(), Some("a"));
        assert_eq!(att_type.to_string(), "(a | b | a)");
        assert!(att_type.is_tokenized());
        assert!(!att_type.is_list());
    }

    #[test]
    fn cdata_is_the_only_untokenized_type() {
        assert!(!AttributeType::Cdata.is_tokenized());
        assert!(AttributeType::NmTokens.is_list());
        assert_eq!(AttributeType::IdRefs.keyword(), "IDREFS");
    }
}
