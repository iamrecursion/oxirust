//! Entity and notation declarations.
//!
//! Mirrors the `EntityDecl` production of XML 1.0 §4.2 `[70]`, the
//! `ExternalID` / `NDataDecl` productions `[75]`, `[76]` and the
//! `NotationDecl` production §4.7 `[82]`.

use core::fmt;

/// An external identifier: a system identifier, optionally preceded by a public
/// one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExternalId {
    public_id: Option<String>,
    system_id: Option<String>,
}

impl ExternalId {
    /// `SYSTEM "system"`.
    #[must_use]
    pub fn system(system_id: impl Into<String>) -> Self {
        Self {
            public_id: None,
            system_id: Some(system_id.into()),
        }
    }

    /// `PUBLIC "public" "system"`.
    #[must_use]
    pub fn public(public_id: impl Into<String>, system_id: impl Into<String>) -> Self {
        Self {
            public_id: Some(public_id.into()),
            system_id: Some(system_id.into()),
        }
    }

    /// `PUBLIC "public"` with no system identifier.
    ///
    /// Only legal in a notation declaration (XML 1.0 §4.7 `[83]`).
    #[must_use]
    pub fn public_only(public_id: impl Into<String>) -> Self {
        Self {
            public_id: Some(public_id.into()),
            system_id: None,
        }
    }

    /// The public identifier, when one was given.
    #[must_use]
    pub fn public_id(&self) -> Option<&str> {
        self.public_id.as_deref()
    }

    /// The system identifier, when one was given.
    #[must_use]
    pub fn system_id(&self) -> Option<&str> {
        self.system_id.as_deref()
    }
}

impl fmt::Display for ExternalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.public_id, &self.system_id) {
            (Some(public), Some(system)) => write!(f, "PUBLIC \"{public}\" \"{system}\""),
            (Some(public), None) => write!(f, "PUBLIC \"{public}\""),
            (None, Some(system)) => write!(f, "SYSTEM \"{system}\""),
            (None, None) => Ok(()),
        }
    }
}

/// Whether an entity is a general entity (`&name;`) or a parameter entity
/// (`%name;`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    /// A general entity, referenced as `&name;` in content and attribute values.
    General,
    /// A parameter entity, referenced as `%name;` inside the DTD.
    Parameter,
}

impl fmt::Display for EntityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::General => "general entity",
            Self::Parameter => "parameter entity",
        })
    }
}

/// What an entity stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityDefinition {
    /// An internal entity: the literal replacement text, with parameter-entity
    /// references and character references already expanded, and general-entity
    /// references left *bypassed* (still spelled `&name;`) exactly as XML 1.0
    /// §4.4.7 requires.
    Internal(String),
    /// An external entity, to be fetched through the
    /// [`EntityResolver`](crate::EntityResolver).
    External {
        /// Where to fetch it from.
        id: ExternalId,
        /// The notation name of an *unparsed* entity (`NDATA name`), or `None`
        /// for an external parsed entity.
        notation: Option<String>,
    },
}

impl EntityDefinition {
    /// Returns `true` for an unparsed entity — one declared with `NDATA`.
    ///
    /// Unparsed entities may only be named by `ENTITY`/`ENTITIES` attributes,
    /// never referenced as `&name;` (XML 1.0 §4.1, WFC: Parsed Entity).
    #[must_use]
    pub const fn is_unparsed(&self) -> bool {
        matches!(
            self,
            Self::External {
                notation: Some(_),
                ..
            }
        )
    }

    /// Returns `true` for an external entity, parsed or not.
    #[must_use]
    pub const fn is_external(&self) -> bool {
        matches!(self, Self::External { .. })
    }
}

/// An entity declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityDecl {
    name: String,
    kind: EntityKind,
    definition: EntityDefinition,
    declared_externally: bool,
    base: Option<String>,
}

impl EntityDecl {
    /// Builds an entity declaration.
    ///
    /// `declared_externally` records whether the declaration itself came from
    /// the external subset or from a parameter entity — the distinction the
    /// standalone-document declaration of XML 1.0 §2.9 turns on.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        kind: EntityKind,
        definition: EntityDefinition,
        declared_externally: bool,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            definition,
            declared_externally,
            base: None,
        }
    }

    /// Records the base URI in force where the declaration appeared, used to
    /// resolve a relative system identifier.
    #[must_use]
    pub fn with_base(mut self, base: Option<String>) -> Self {
        self.base = base;
        self
    }

    /// The entity name, without the `&`/`%` and `;`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this is a general or a parameter entity.
    #[must_use]
    pub fn kind(&self) -> EntityKind {
        self.kind
    }

    /// What the entity stands for.
    #[must_use]
    pub fn definition(&self) -> &EntityDefinition {
        &self.definition
    }

    /// Whether the declaration came from outside the internal subset.
    #[must_use]
    pub fn is_declared_externally(&self) -> bool {
        self.declared_externally
    }

    /// The base URI in force where the declaration appeared.
    #[must_use]
    pub fn base(&self) -> Option<&str> {
        self.base.as_deref()
    }

    /// Shorthand for [`EntityDefinition::is_unparsed`].
    #[must_use]
    pub const fn is_unparsed(&self) -> bool {
        self.definition.is_unparsed()
    }

    /// The notation name of an unparsed entity.
    #[must_use]
    pub fn notation(&self) -> Option<&str> {
        match &self.definition {
            EntityDefinition::External { notation, .. } => notation.as_deref(),
            EntityDefinition::Internal(_) => None,
        }
    }

    /// The replacement text of an internal entity.
    #[must_use]
    pub fn internal_text(&self) -> Option<&str> {
        match &self.definition {
            EntityDefinition::Internal(text) => Some(text),
            EntityDefinition::External { .. } => None,
        }
    }
}

impl fmt::Display for EntityDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<!ENTITY ")?;
        if self.kind == EntityKind::Parameter {
            f.write_str("% ")?;
        }
        write!(f, "{}", self.name)?;
        match &self.definition {
            EntityDefinition::Internal(text) => write!(f, " \"{text}\">"),
            EntityDefinition::External { id, notation } => {
                write!(f, " {id}")?;
                if let Some(notation) = notation {
                    write!(f, " NDATA {notation}")?;
                }
                f.write_str(">")
            }
        }
    }
}

/// A notation declaration, `<!NOTATION name ExternalID | PublicID>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotationDecl {
    name: String,
    id: ExternalId,
    declared_externally: bool,
}

impl NotationDecl {
    /// Builds a notation declaration.
    #[must_use]
    pub fn new(name: impl Into<String>, id: ExternalId, declared_externally: bool) -> Self {
        Self {
            name: name.into(),
            id,
            declared_externally,
        }
    }

    /// The notation name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The external identifier that locates the notation's specification.
    #[must_use]
    pub fn id(&self) -> &ExternalId {
        &self.id
    }

    /// The public identifier, when one was given.
    #[must_use]
    pub fn public_id(&self) -> Option<&str> {
        self.id.public_id()
    }

    /// The system identifier, when one was given.
    #[must_use]
    pub fn system_id(&self) -> Option<&str> {
        self.id.system_id()
    }

    /// Whether the declaration came from outside the internal subset.
    #[must_use]
    pub fn is_declared_externally(&self) -> bool {
        self.declared_externally
    }
}

impl fmt::Display for NotationDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<!NOTATION {} {}>", self.name, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unparsed_entities_are_recognised() {
        let entity = EntityDecl::new(
            "logo",
            EntityKind::General,
            EntityDefinition::External {
                id: ExternalId::system("logo.gif"),
                notation: Some("gif".to_owned()),
            },
            false,
        );
        assert!(entity.is_unparsed());
        assert_eq!(entity.notation(), Some("gif"));
        assert_eq!(entity.internal_text(), None);
        assert_eq!(
            entity.to_string(),
            "<!ENTITY logo SYSTEM \"logo.gif\" NDATA gif>"
        );
    }

    #[test]
    fn parameter_entity_display_includes_the_percent_sign() {
        let entity = EntityDecl::new(
            "p",
            EntityKind::Parameter,
            EntityDefinition::Internal("<!ELEMENT a EMPTY>".to_owned()),
            true,
        );
        assert_eq!(entity.to_string(), "<!ENTITY % p \"<!ELEMENT a EMPTY>\">");
        assert!(entity.is_declared_externally());
    }

    #[test]
    fn external_id_display_matches_the_grammar() {
        assert_eq!(ExternalId::system("a.dtd").to_string(), "SYSTEM \"a.dtd\"");
        assert_eq!(
            ExternalId::public("-//X//DTD//EN", "a.dtd").to_string(),
            "PUBLIC \"-//X//DTD//EN\" \"a.dtd\""
        );
        assert_eq!(
            ExternalId::public_only("-//X//N//EN").to_string(),
            "PUBLIC \"-//X//N//EN\""
        );
    }
}
