//! Errors reported while reading a document type definition.
//!
//! Two error families live here:
//!
//! * [`DtdError`] — a *fatal* error: the DTD text could not be read at all
//!   (a syntax error, an undeclared parameter entity, a blown expansion
//!   budget, a denied external entity, …). XML 1.0 §5.1 calls these fatal
//!   errors, and a conforming processor must not continue normal processing
//!   after one.
//! * [`ValidityError`] — a *validity* error: the DTD was read fine, but a
//!   document violates one of the validity constraints of XML 1.0 §3–§4.
//!   A non-validating processor is allowed to ignore these; a validating one
//!   must report them. See [`crate::validate`].

use core::fmt;

use oxixml_support::pos::TextPosition;

/// Which of the [entity-expansion limits](crate::ExpansionLimits) was exceeded.
///
/// Each variant carries the configured limit that the input ran into, so that
/// a caller can tell a genuinely huge (but honest) document apart from an
/// exponential-expansion attack, and raise the relevant knob if wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpansionLimit {
    /// The nesting depth of entity references exceeded
    /// [`ExpansionLimits::max_depth`](crate::ExpansionLimits::max_depth).
    ///
    /// This is the guard that stops a self-referential entity (which is also
    /// a well-formedness error, but one that a naive expander only discovers
    /// by running out of stack).
    Depth(usize),
    /// The number of individual entity expansions exceeded
    /// [`ExpansionLimits::max_expansions`](crate::ExpansionLimits::max_expansions).
    ///
    /// This is the guard that stops the classic "billion laughs" bomb, where
    /// ten mutually nested entities expand to `10^9` characters through only
    /// a few hundred bytes of declarations.
    Count(usize),
    /// The total number of characters produced by entity expansion exceeded
    /// [`ExpansionLimits::max_expanded_size`](crate::ExpansionLimits::max_expanded_size).
    ///
    /// This is the guard that stops quadratic blow-up, where a single large
    /// entity is referenced many times.
    TotalSize(usize),
}

impl fmt::Display for ExpansionLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Depth(limit) => write!(f, "entity nesting depth limit of {limit} exceeded"),
            Self::Count(limit) => write!(f, "entity expansion count limit of {limit} exceeded"),
            Self::TotalSize(limit) => {
                write!(
                    f,
                    "total entity expansion size limit of {limit} characters exceeded"
                )
            }
        }
    }
}

/// The reason a DTD could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DtdErrorKind {
    /// The input ended in the middle of a construct that was still open.
    UnexpectedEof {
        /// What the parser was still waiting for, e.g. `">"` or `"']]>'"`.
        expected: String,
    },
    /// A character that the grammar does not allow at this point.
    Unexpected {
        /// What the grammar allows here, e.g. `"'#PCDATA' or a name"`.
        expected: String,
        /// What was actually found, already rendered for humans.
        found: String,
    },
    /// A name did not match the `Name` production of XML 1.0 §2.3 `[5]`.
    InvalidName(String),
    /// A token did not match the `Nmtoken` production of XML 1.0 §2.3 `[7]`.
    InvalidNmtoken(String),
    /// A public identifier contained a character outside `PubidChar`
    /// (XML 1.0 §2.3 `[13]`).
    InvalidPublicId(String),
    /// A character reference did not denote a legal XML `Char`.
    InvalidCharRef(String),
    /// A parameter entity was referenced but never declared.
    ///
    /// For a validating processor this is a fatal error (XML 1.0 §4.1, WFC:
    /// Entity Declared, plus §4.4.1).
    UndeclaredParameterEntity(String),
    /// A general entity was referenced but never declared.
    UndeclaredGeneralEntity(String),
    /// An entity was (directly or indirectly) defined in terms of itself,
    /// violating the "No Recursion" well-formedness constraint of XML 1.0 §4.1.
    RecursiveEntity {
        /// The entity that closed the cycle.
        name: String,
        /// The chain of entity names that led back to it, outermost first.
        chain: Vec<String>,
    },
    /// One of the [expansion limits](ExpansionLimit) was exceeded.
    ///
    /// The DTD is syntactically fine but expanding it would cost more than the
    /// configured budget allows — the guard against entity-expansion denial of
    /// service.
    EntityExpansionLimit(ExpansionLimit),
    /// An external entity had to be read, but the configured
    /// [`EntityResolver`](crate::EntityResolver) refused to supply it.
    ///
    /// This is the default behaviour: [`DenyExternalEntities`](crate::DenyExternalEntities)
    /// resolves nothing, so no network or file access can happen unless the
    /// caller opts in explicitly.
    ExternalEntityDenied {
        /// The system identifier that would have been dereferenced.
        system_id: String,
    },
    /// The configured [`EntityResolver`](crate::EntityResolver) tried to supply
    /// an external entity and failed.
    ExternalEntityUnavailable {
        /// The system identifier that could not be dereferenced.
        system_id: String,
        /// The resolver's explanation.
        reason: String,
    },
    /// A parameter-entity reference appeared inside a markup declaration in the
    /// *internal* subset, violating the "PEs in Internal Subset" well-formedness
    /// constraint of XML 1.0 §2.8.
    ParameterEntityInInternalSubset(String),
    /// A general entity was referenced in an entity value or attribute default
    /// but resolves to an *unparsed* entity, violating the "Parsed Entity"
    /// well-formedness constraint of XML 1.0 §4.1.
    UnparsedEntityReference(String),
    /// A conditional section used a keyword other than `INCLUDE` or `IGNORE`.
    BadConditionalSection(String),
    /// A conditional section appeared in the internal subset, where XML 1.0
    /// §3.4 `[28]` does not permit it.
    ConditionalSectionInInternalSubset,
    /// WFC: No `<` in Attribute Values (XML 1.0 §3.1) — the replacement text of
    /// an entity referred to directly or indirectly from an attribute value
    /// contains a `<`.
    ///
    /// A `<` written literally in the attribute value is rejected where the
    /// value is scanned; this is the *indirect* half of the constraint, which
    /// only a processor that expands entities can see.
    LessThanInAttributeValue {
        /// The chain of entities being expanded, outermost first; the last
        /// entry is the entity whose replacement text carries the `<`.
        chain: Vec<String>,
    },
    /// The parenthesised groups of an element content model (XML 1.0 §3.2.1
    /// `[47]`–`[50]`) nest deeper than the parser's cap,
    /// [`MAX_CONTENT_MODEL_DEPTH`](crate::MAX_CONTENT_MODEL_DEPTH).
    ///
    /// Like the entity-expansion budget, this turns a hostile declaration —
    /// `<!ELEMENT e ((((…a…))))>` nested tens of thousands deep — into a clean
    /// error instead of a stack overflow.
    ContentModelDepth(usize),
    /// A `children` content model (XML 1.0 §3.2.1 `[47]`) names more child-element
    /// positions than the parser's cap,
    /// [`MAX_CONTENT_MODEL_POSITIONS`](crate::MAX_CONTENT_MODEL_POSITIONS).
    ///
    /// Compiling a content model into the automaton of [`crate::automaton`] costs
    /// superlinear time in the number of distinct leaf names inside one choice or sequence
    /// group, entirely independent of parenthesis-nesting depth — a hostile
    /// `<!ELEMENT e (e0|e1|…|e9999)+>` is one parenthesis deep but still costs seconds to
    /// compile. This cap turns that into a clean error before the expensive automaton
    /// construction runs at all, the same way
    /// [`ContentModelDepth`](Self::ContentModelDepth) guards against pathological nesting.
    ContentModelPositions(usize),
}

impl fmt::Display for DtdErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { expected } => {
                write!(f, "unexpected end of input, expected {expected}")
            }
            Self::Unexpected { expected, found } => {
                write!(f, "expected {expected}, found {found}")
            }
            Self::InvalidName(name) => write!(f, "`{name}` is not a valid XML Name"),
            Self::InvalidNmtoken(token) => write!(f, "`{token}` is not a valid Nmtoken"),
            Self::InvalidPublicId(id) => {
                write!(f, "`{id}` is not a valid public identifier")
            }
            Self::InvalidCharRef(text) => {
                write!(f, "`&#{text};` is not a valid character reference")
            }
            Self::UndeclaredParameterEntity(name) => {
                write!(f, "parameter entity `%{name};` is not declared")
            }
            Self::UndeclaredGeneralEntity(name) => {
                write!(f, "general entity `&{name};` is not declared")
            }
            Self::RecursiveEntity { name, chain } => {
                write!(
                    f,
                    "entity `{name}` is recursive (via {})",
                    chain.join(" -> ")
                )
            }
            Self::EntityExpansionLimit(limit) => write!(f, "{limit}"),
            Self::ExternalEntityDenied { system_id } => write!(
                f,
                "resolution of external entity `{system_id}` was denied by the entity resolver"
            ),
            Self::ExternalEntityUnavailable { system_id, reason } => {
                write!(
                    f,
                    "external entity `{system_id}` could not be read: {reason}"
                )
            }
            Self::ParameterEntityInInternalSubset(name) => write!(
                f,
                "parameter entity `%{name};` may not occur inside a markup declaration \
                 in the internal subset"
            ),
            Self::UnparsedEntityReference(name) => {
                write!(f, "`&{name};` refers to an unparsed entity")
            }
            Self::BadConditionalSection(keyword) => {
                write!(f, "`{keyword}` is not a conditional-section keyword")
            }
            Self::ConditionalSectionInInternalSubset => {
                write!(
                    f,
                    "conditional sections are not allowed in the internal subset"
                )
            }
            Self::LessThanInAttributeValue { chain } => write!(
                f,
                "the replacement text of entity `{}` contains a `<`, which an attribute value \
                 may not contain",
                chain.join(" -> ")
            ),
            Self::ContentModelDepth(limit) => write!(
                f,
                "the content model nests parenthesised groups more than {limit} levels deep"
            ),
            Self::ContentModelPositions(limit) => write!(
                f,
                "the content model names more than {limit} child-element positions"
            ),
        }
    }
}

/// A fatal error encountered while reading a DTD.
///
/// Carries the [kind](DtdErrorKind), the position in the *current* entity, and
/// the stack of entity names that were being expanded — which is what makes an
/// error inside a deeply nested parameter entity actionable.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{DtdErrorKind, DtdParser};
///
/// let error = DtdParser::new()
///     .parse_internal_subset("<!ELEMENT a (b|)>")
///     .expect_err("a choice list may not end with `|`");
/// assert!(matches!(error.kind(), DtdErrorKind::InvalidName(_)));
/// assert_eq!(error.entity_stack(), &[] as &[String]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtdError {
    kind: DtdErrorKind,
    position: TextPosition,
    entity_stack: Vec<String>,
}

impl DtdError {
    /// Builds an error at `position`, expanded through `entity_stack`
    /// (outermost entity first).
    #[must_use]
    pub fn new(kind: DtdErrorKind, position: TextPosition, entity_stack: Vec<String>) -> Self {
        Self {
            kind,
            position,
            entity_stack,
        }
    }

    /// Builds an error with no position information.
    #[must_use]
    pub fn detached(kind: DtdErrorKind) -> Self {
        Self {
            kind,
            position: TextPosition::start(),
            entity_stack: Vec::new(),
        }
    }

    /// The reason the DTD could not be read.
    #[must_use]
    pub fn kind(&self) -> &DtdErrorKind {
        &self.kind
    }

    /// Consumes the error, yielding its [kind](DtdErrorKind).
    #[must_use]
    pub fn into_kind(self) -> DtdErrorKind {
        self.kind
    }

    /// Where in the innermost entity the error was detected.
    ///
    /// Its `line` and `column` are **0-based** (see [`oxixml_support::pos::TextPosition`]);
    /// only its `Display` impl renders the human-facing 1-based numbering.
    #[must_use]
    pub fn position(&self) -> TextPosition {
        self.position
    }

    /// The chain of entities being expanded when the error was detected,
    /// outermost first. Empty when the error is in the subset text itself.
    #[must_use]
    pub fn entity_stack(&self) -> &[String] {
        &self.entity_stack
    }

    /// Returns `true` when this error is an [`DtdErrorKind::EntityExpansionLimit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_dtd::{DtdError, DtdErrorKind, ExpansionLimit};
    ///
    /// let error = DtdError::detached(DtdErrorKind::EntityExpansionLimit(
    ///     ExpansionLimit::Count(64),
    /// ));
    /// assert!(error.is_expansion_limit());
    /// ```
    #[must_use]
    pub fn is_expansion_limit(&self) -> bool {
        matches!(self.kind, DtdErrorKind::EntityExpansionLimit(_))
    }
}

impl fmt::Display for DtdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if self.entity_stack.is_empty() {
            write!(
                f,
                " (at line {}, column {})",
                self.position.line + 1,
                self.position.column + 1
            )
        } else {
            write!(
                f,
                " (at line {}, column {} of entity `{}`)",
                self.position.line + 1,
                self.position.column + 1,
                self.entity_stack.join("` in `")
            )
        }
    }
}

impl std::error::Error for DtdError {}

/// The validity constraint that a document violated.
///
/// The variant names follow the "Validity constraint:" headings of the XML 1.0
/// Recommendation so that a reader can look each one up directly.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidityErrorKind {
    /// VC: Root Element Type (§2.8) — the document element does not match the
    /// name in the document type declaration.
    RootElementType {
        /// The name given in `<!DOCTYPE …>`.
        declared: String,
        /// The name of the actual document element.
        found: String,
    },
    /// VC: Element Valid (§3) — the element type is not declared at all.
    ElementNotDeclared(String),
    /// VC: Element Valid (§3) — an `EMPTY` element has children or character
    /// data.
    EmptyElementNotEmpty {
        /// The element type.
        element: String,
    },
    /// VC: Element Valid (§3) — character data appeared in element content.
    CharacterDataNotAllowed {
        /// The element type whose content model forbids character data.
        element: String,
    },
    /// VC: Element Valid (§3) — the sequence of children does not match the
    /// content model.
    ContentModelMismatch {
        /// The element type being checked.
        element: String,
        /// The child element that could not be accepted, or `None` when the
        /// element ended too early.
        found: Option<String>,
        /// The child element types the content model would have accepted here.
        expected: Vec<String>,
    },
    /// VC: Attribute Value Type (§3.1) — the attribute is not declared for this
    /// element type.
    AttributeNotDeclared {
        /// The element type carrying the attribute.
        element: String,
        /// The attribute name.
        attribute: String,
    },
    /// VC: Required Attribute (§3.3.2) — a `#REQUIRED` attribute is absent.
    RequiredAttributeMissing {
        /// The element type.
        element: String,
        /// The missing attribute.
        attribute: String,
    },
    /// VC: Fixed Attribute Default (§3.3.2) — a `#FIXED` attribute was given a
    /// value other than its default.
    FixedAttributeMismatch {
        /// The element type.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The declared, fixed value.
        expected: String,
        /// The value that was supplied instead.
        found: String,
    },
    /// VC: ID / IDREF / Entity Name / Name Token (§3.3.1) — the attribute value
    /// is not lexically valid for its declared type.
    AttributeValueInvalid {
        /// The element type.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The offending value.
        value: String,
        /// Why it is not valid for the declared type.
        reason: String,
    },
    /// VC: ID (§3.3.1) — the same `ID` value appeared twice.
    DuplicateId {
        /// The duplicated value.
        value: String,
        /// The element type on which the duplicate occurred.
        element: String,
    },
    /// VC: One ID per Element Type (§3.3.1) — an element type declares more
    /// than one attribute of type `ID`.
    MultipleIdAttributes {
        /// The element type.
        element: String,
    },
    /// VC: ID Attribute Default (§3.3.1) — an `ID` attribute was declared
    /// `#FIXED` or with a literal default.
    IdAttributeHasDefault {
        /// The element type.
        element: String,
        /// The attribute name.
        attribute: String,
    },
    /// VC: IDREF (§3.3.1) — an `IDREF`/`IDREFS` value does not match any `ID`
    /// in the document.
    UnresolvedIdref {
        /// The dangling reference.
        value: String,
    },
    /// VC: Entity Declared (§4.1) — a general entity is referenced but never
    /// declared. (In a document with no external subset and no parameter
    /// entities this is a well-formedness error instead.)
    EntityNotDeclared {
        /// The undeclared entity.
        name: String,
    },
    /// VC: Entity Name (§3.3.1) — an `ENTITY`/`ENTITIES` value does not name a
    /// declared unparsed entity.
    UnknownUnparsedEntity {
        /// The entity name that was referenced.
        value: String,
    },
    /// VC: Notation Attributes (§3.3.1) / Notation Declared (§4.7) — a notation
    /// name is not declared.
    NotationNotDeclared {
        /// The notation name.
        value: String,
    },
    /// VC: No Duplicate Types (§3.2.2) — a mixed content model lists the same
    /// element type twice.
    DuplicateTypeInMixedContent {
        /// The element whose content model is at fault.
        element: String,
        /// The duplicated child element type.
        duplicate: String,
    },
    /// VC: Unique Element Type Declaration (§3) — the same element type is
    /// declared more than once.
    DuplicateElementDeclaration {
        /// The element type declared twice.
        element: String,
    },
    /// VC: No Notation on Empty Element (§3.3.1) — an `EMPTY` element declares
    /// a `NOTATION` attribute.
    NotationOnEmptyElement {
        /// The element type.
        element: String,
    },
    /// VC: One Notation per Element Type (§3.3.1) — an element type declares
    /// more than one attribute of type `NOTATION`.
    MultipleNotationAttributes {
        /// The element type.
        element: String,
    },
    /// VC: No Duplicate Tokens (§3.3.1) — a `NOTATION` or enumeration
    /// declaration lists the same token twice.
    DuplicateEnumerationToken {
        /// The element type.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The duplicated token.
        token: String,
    },
    /// VC: Standalone Document Declaration (§2.9) — a document declared
    /// `standalone="yes"` depends on markup in the external subset.
    StandaloneViolation {
        /// A human-readable description of the dependency.
        detail: String,
    },
    /// VC: Proper Conditional Section/PE Nesting (§3.4) — the `<![`, `[` and
    /// `]]>` of a conditional section do not all come from the same entity.
    ImproperConditionalSectionNesting,
    /// VC: Proper Declaration/PE Nesting (§2.8) — a markup declaration begins
    /// in one entity and ends in another.
    ///
    /// Note that this is a *validity* error and not a fatal one, even though
    /// several widely used parsers treat it as fatal.
    ImproperDeclarationNesting {
        /// The entity whose replacement text finished the declaration.
        entity: String,
    },
    /// VC: Proper Group/PE Nesting (§3.2.1) — a parenthesised group in a
    /// content model begins in one entity and ends in another.
    ImproperGroupNesting {
        /// The element type whose content model is at fault.
        element: String,
    },
    /// A deterministic-content-model requirement (§3.2.1, Appendix E) — the
    /// content model is ambiguous.
    NonDeterministicContentModel {
        /// The element type whose content model is ambiguous.
        element: String,
    },
    /// VC: Notation Declared (§4.7) — an unparsed entity names an undeclared
    /// notation.
    EntityNotationNotDeclared {
        /// The unparsed entity.
        entity: String,
        /// The notation it names.
        notation: String,
    },
    /// VC: Attribute Default Value Syntactically Correct (§3.3.2) — the
    /// declared default value is not valid for the attribute's declared type.
    DefaultValueInvalid {
        /// The element type.
        element: String,
        /// The attribute name.
        attribute: String,
        /// The offending default value.
        value: String,
        /// Why it is not valid.
        reason: String,
    },
}

impl fmt::Display for ValidityErrorKind {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootElementType { declared, found } => write!(
                f,
                "the document element is `{found}` but the document type declaration names \
                 `{declared}`"
            ),
            Self::ElementNotDeclared(name) => {
                write!(f, "element type `{name}` is not declared")
            }
            Self::EmptyElementNotEmpty { element } => {
                write!(f, "element `{element}` is declared EMPTY but has content")
            }
            Self::CharacterDataNotAllowed { element } => write!(
                f,
                "character data is not allowed in the element content of `{element}`"
            ),
            Self::ContentModelMismatch {
                element,
                found,
                expected,
            } => {
                write!(
                    f,
                    "the content of `{element}` does not match its content model: "
                )?;
                match found {
                    Some(name) => write!(f, "`{name}` is not allowed here")?,
                    None => write!(f, "the element ends too early")?,
                }
                if expected.is_empty() {
                    Ok(())
                } else {
                    write!(f, " (expected one of: {})", expected.join(", "))
                }
            }
            Self::AttributeNotDeclared { element, attribute } => {
                write!(
                    f,
                    "attribute `{attribute}` is not declared for element `{element}`"
                )
            }
            Self::RequiredAttributeMissing { element, attribute } => write!(
                f,
                "required attribute `{attribute}` is missing on element `{element}`"
            ),
            Self::FixedAttributeMismatch {
                element,
                attribute,
                expected,
                found,
            } => write!(
                f,
                "attribute `{attribute}` of `{element}` is #FIXED \"{expected}\" but was given \
                 \"{found}\""
            ),
            Self::AttributeValueInvalid {
                element,
                attribute,
                value,
                reason,
            } => write!(
                f,
                "value \"{value}\" of attribute `{attribute}` on `{element}` is invalid: {reason}"
            ),
            Self::DuplicateId { value, element } => {
                write!(f, "ID value \"{value}\" on `{element}` is already used")
            }
            Self::MultipleIdAttributes { element } => {
                write!(f, "element `{element}` declares more than one ID attribute")
            }
            Self::IdAttributeHasDefault { element, attribute } => write!(
                f,
                "ID attribute `{attribute}` of `{element}` must be #IMPLIED or #REQUIRED"
            ),
            Self::UnresolvedIdref { value } => {
                write!(f, "IDREF \"{value}\" does not match any ID in the document")
            }
            Self::EntityNotDeclared { name } => {
                write!(f, "general entity `{name}` is not declared")
            }
            Self::UnknownUnparsedEntity { value } => {
                write!(f, "\"{value}\" is not a declared unparsed entity")
            }
            Self::NotationNotDeclared { value } => {
                write!(f, "notation `{value}` is not declared")
            }
            Self::DuplicateTypeInMixedContent { element, duplicate } => write!(
                f,
                "the mixed content model of `{element}` lists `{duplicate}` more than once"
            ),
            Self::DuplicateElementDeclaration { element } => {
                write!(f, "element type `{element}` is declared more than once")
            }
            Self::NotationOnEmptyElement { element } => write!(
                f,
                "element `{element}` is declared EMPTY and so may not have a NOTATION attribute"
            ),
            Self::MultipleNotationAttributes { element } => {
                write!(
                    f,
                    "element `{element}` declares more than one NOTATION attribute"
                )
            }
            Self::DuplicateEnumerationToken {
                element,
                attribute,
                token,
            } => write!(
                f,
                "token `{token}` occurs more than once in the declaration of `{attribute}` \
                 on `{element}`"
            ),
            Self::StandaloneViolation { detail } => {
                write!(f, "standalone=\"yes\" is violated: {detail}")
            }
            Self::ImproperDeclarationNesting { entity } => write!(
                f,
                "a markup declaration is finished by the replacement text of parameter entity \
                 `%{entity};` rather than by the entity it started in"
            ),
            Self::ImproperConditionalSectionNesting => write!(
                f,
                "the `<![`, `[` and `]]>` of a conditional section do not all come from the \
                 same parameter entity"
            ),
            Self::ImproperGroupNesting { element } => write!(
                f,
                "a group in the content model of `{element}` starts and ends in different \
                 parameter entities"
            ),
            Self::NonDeterministicContentModel { element } => {
                write!(f, "the content model of `{element}` is not deterministic")
            }
            Self::EntityNotationNotDeclared { entity, notation } => write!(
                f,
                "unparsed entity `{entity}` names notation `{notation}`, which is not declared"
            ),
            Self::DefaultValueInvalid {
                element,
                attribute,
                value,
                reason,
            } => write!(
                f,
                "the default value \"{value}\" of attribute `{attribute}` on `{element}` is \
                 invalid: {reason}"
            ),
        }
    }
}

/// A violated validity constraint, together with where it was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityError {
    kind: ValidityErrorKind,
    position: TextPosition,
}

impl ValidityError {
    /// Builds a validity error at `position`.
    #[must_use]
    pub fn new(kind: ValidityErrorKind, position: TextPosition) -> Self {
        Self { kind, position }
    }

    /// Builds a validity error with no position information.
    #[must_use]
    pub fn detached(kind: ValidityErrorKind) -> Self {
        Self {
            kind,
            position: TextPosition::start(),
        }
    }

    /// The constraint that was violated.
    #[must_use]
    pub fn kind(&self) -> &ValidityErrorKind {
        &self.kind
    }

    /// Consumes the error, yielding the violated constraint.
    #[must_use]
    pub fn into_kind(self) -> ValidityErrorKind {
        self.kind
    }

    /// Where the violation was detected.
    ///
    /// Its `line` and `column` are **0-based** (see [`oxixml_support::pos::TextPosition`]);
    /// only its `Display` impl renders the human-facing 1-based numbering.
    #[must_use]
    pub fn position(&self) -> TextPosition {
        self.position
    }
}

impl fmt::Display for ValidityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (at line {}, column {})",
            self.kind,
            self.position.line + 1,
            self.position.column + 1
        )
    }
}

impl std::error::Error for ValidityError {}

/// Either kind of failure that reading *and* validating a document can produce.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The document was not well formed, or the DTD could not be read.
    ///
    /// Boxed to keep `Result<_, Error>` small: an unboxed [`DtdError`] would
    /// make every success path pay for the error path's size.
    Dtd(Box<DtdError>),
    /// The document was well formed but not valid.
    Validity(Box<ValidityError>),
    /// The XML document itself could not be parsed.
    Xml(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dtd(error) => write!(f, "{error}"),
            Self::Validity(error) => write!(f, "{error}"),
            Self::Xml(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<DtdError> for Error {
    fn from(error: DtdError) -> Self {
        Self::Dtd(Box::new(error))
    }
}

impl From<ValidityError> for Error {
    fn from(error: ValidityError) -> Self {
        Self::Validity(Box::new(error))
    }
}
