//! The compiled document type definition.
//!
//! [`Dtd`] is the whole DTD — internal subset, external subset and everything
//! reached through parameter entities — after parsing: element type
//! declarations with their content models already compiled into automata,
//! merged attribute lists, entities and notations.
//!
//! It is a plain data structure: build one with [`DtdParser`](crate::DtdParser)
//! or assemble one by hand, then hand it to
//! [`Validator`](crate::validate::Validator).

mod attribute;
mod element;
mod entity;

pub use attribute::{AttributeDecl, AttributeList, AttributeType, DefaultDecl};
pub use element::{ContentParticle, ContentSpec, ElementDecl, MixedContent, Occurrence};
pub use entity::{EntityDecl, EntityDefinition, EntityKind, ExternalId, NotationDecl};

use std::collections::BTreeMap;

use crate::automaton::CompiledContent;
use crate::error::{ValidityError, ValidityErrorKind};

/// A parsed document type definition.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::DtdParser;
///
/// let dtd = DtdParser::new().parse_internal_subset(
///     r#"<!ELEMENT list (item+)>
///        <!ELEMENT item (#PCDATA)>
///        <!ATTLIST item id ID #REQUIRED>
///        <!ENTITY ship "&#x1F6A2;">
///        <!NOTATION gif SYSTEM "image/gif">"#,
/// )?;
///
/// assert_eq!(dtd.elements().count(), 2);
/// assert!(dtd.attributes("item").and_then(|list| list.get("id")).is_some());
/// assert_eq!(dtd.general_entity("ship").and_then(|e| e.internal_text()), Some("\u{1F6A2}"));
/// assert!(dtd.notation("gif").is_some());
/// # Ok::<(), oxixml_dtd::DtdError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dtd {
    name: Option<String>,
    external_id: Option<ExternalId>,
    elements: BTreeMap<String, ElementDecl>,
    compiled: BTreeMap<String, CompiledContent>,
    attributes: BTreeMap<String, AttributeList>,
    general_entities: BTreeMap<String, EntityDecl>,
    parameter_entities: BTreeMap<String, EntityDecl>,
    notations: BTreeMap<String, NotationDecl>,
    declaration_errors: Vec<ValidityError>,
    has_external_subset: bool,
    saw_parameter_entity_reference: bool,
    external_subset_read: bool,
}

impl Dtd {
    /// Creates an empty DTD.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The element type name given in the document type declaration, when the
    /// DTD was read from one.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Sets the document element type name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// The external identifier of the external subset, if the document type
    /// declaration named one.
    #[must_use]
    pub fn external_id(&self) -> Option<&ExternalId> {
        self.external_id.as_ref()
    }

    /// Records the external subset's identifier.
    pub fn set_external_id(&mut self, id: ExternalId) {
        self.external_id = Some(id);
        self.has_external_subset = true;
    }

    /// Whether the document type declaration named an external subset.
    #[must_use]
    pub fn has_external_subset(&self) -> bool {
        self.has_external_subset
    }

    /// Whether the external subset was actually read (as opposed to named but
    /// left unresolved).
    #[must_use]
    pub fn external_subset_read(&self) -> bool {
        self.external_subset_read
    }

    /// Records that the external subset was read.
    pub fn set_external_subset_read(&mut self, read: bool) {
        self.external_subset_read = read;
    }

    /// Whether any parameter-entity reference was expanded while reading. A
    /// `standalone="yes"` document may not depend on such markup (XML 1.0 §2.9).
    #[must_use]
    pub fn saw_parameter_entity_reference(&self) -> bool {
        self.saw_parameter_entity_reference
    }

    /// Records that a parameter-entity reference was expanded.
    pub fn note_parameter_entity_reference(&mut self) {
        self.saw_parameter_entity_reference = true;
    }

    /// Looks up an element type declaration.
    #[must_use]
    pub fn element(&self, name: &str) -> Option<&ElementDecl> {
        self.elements.get(name)
    }

    /// Iterates over every element type declaration, in name order.
    pub fn elements(&self) -> impl Iterator<Item = &ElementDecl> {
        self.elements.values()
    }

    /// The compiled content model of an element type.
    #[must_use]
    pub fn compiled_content(&self, name: &str) -> Option<&CompiledContent> {
        self.compiled.get(name)
    }

    /// Adds an element type declaration.
    ///
    /// XML 1.0 §3 makes a second declaration of the same element type a
    /// validity error; the first declaration is kept and the violation is
    /// recorded in [`declaration_errors`](Self::declaration_errors).
    ///
    /// Returns `true` when the declaration was kept.
    pub fn add_element(&mut self, declaration: ElementDecl) -> bool {
        if self.elements.contains_key(declaration.name()) {
            self.declaration_errors.push(ValidityError::detached(
                ValidityErrorKind::DuplicateElementDeclaration {
                    element: declaration.name().to_owned(),
                },
            ));
            return false;
        }
        let compiled = CompiledContent::compile(declaration.content());
        if let ContentSpec::Mixed(mixed) = declaration.content()
            && let Some(duplicate) = mixed.first_duplicate()
        {
            self.declaration_errors.push(ValidityError::detached(
                ValidityErrorKind::DuplicateTypeInMixedContent {
                    element: declaration.name().to_owned(),
                    duplicate: duplicate.to_owned(),
                },
            ));
        }
        if !compiled.is_deterministic() {
            self.declaration_errors.push(ValidityError::detached(
                ValidityErrorKind::NonDeterministicContentModel {
                    element: declaration.name().to_owned(),
                },
            ));
        }
        self.compiled
            .insert(declaration.name().to_owned(), compiled);
        self.elements
            .insert(declaration.name().to_owned(), declaration);
        true
    }

    /// The attributes declared for an element type.
    #[must_use]
    pub fn attributes(&self, element: &str) -> Option<&AttributeList> {
        self.attributes.get(element)
    }

    /// Iterates over every attribute list, in element-name order.
    pub fn attribute_lists(&self) -> impl Iterator<Item = &AttributeList> {
        self.attributes.values()
    }

    /// Mutable access to an element type's attribute list.
    pub(crate) fn attributes_mut(&mut self, element: &str) -> Option<&mut AttributeList> {
        self.attributes.get_mut(element)
    }

    /// Adds one attribute definition to an element type's list, applying the
    /// first-declaration-wins rule of XML 1.0 §3.3.
    ///
    /// Returns `true` when the definition was kept.
    pub fn add_attribute(&mut self, element: &str, declaration: AttributeDecl) -> bool {
        self.attributes
            .entry(element.to_owned())
            .or_insert_with(|| AttributeList::new(element))
            .insert(declaration)
    }

    /// Looks up a general entity declaration.
    #[must_use]
    pub fn general_entity(&self, name: &str) -> Option<&EntityDecl> {
        self.general_entities.get(name)
    }

    /// Iterates over every general entity declaration, in name order.
    pub fn general_entities(&self) -> impl Iterator<Item = &EntityDecl> {
        self.general_entities.values()
    }

    /// Looks up a parameter entity declaration.
    #[must_use]
    pub fn parameter_entity(&self, name: &str) -> Option<&EntityDecl> {
        self.parameter_entities.get(name)
    }

    /// Iterates over every parameter entity declaration, in name order.
    pub fn parameter_entities(&self) -> impl Iterator<Item = &EntityDecl> {
        self.parameter_entities.values()
    }

    /// Adds an entity declaration.
    ///
    /// XML 1.0 §4.2 says that when an entity is declared more than once, the
    /// *first* declaration binds; the later ones are discarded without error.
    ///
    /// Returns `true` when the declaration was kept.
    pub fn add_entity(&mut self, declaration: EntityDecl) -> bool {
        let table = match declaration.kind() {
            EntityKind::General => &mut self.general_entities,
            EntityKind::Parameter => &mut self.parameter_entities,
        };
        if table.contains_key(declaration.name()) {
            return false;
        }
        table.insert(declaration.name().to_owned(), declaration);
        true
    }

    /// Looks up a notation declaration.
    #[must_use]
    pub fn notation(&self, name: &str) -> Option<&NotationDecl> {
        self.notations.get(name)
    }

    /// Iterates over every notation declaration, in name order.
    pub fn notations(&self) -> impl Iterator<Item = &NotationDecl> {
        self.notations.values()
    }

    /// Adds a notation declaration. A repeated declaration of the same notation
    /// is discarded.
    ///
    /// Returns `true` when the declaration was kept.
    pub fn add_notation(&mut self, declaration: NotationDecl) -> bool {
        if self.notations.contains_key(declaration.name()) {
            return false;
        }
        self.notations
            .insert(declaration.name().to_owned(), declaration);
        true
    }

    /// Validity errors that are properties of the DTD itself rather than of any
    /// particular document — duplicate element declarations, duplicate tokens
    /// in an enumeration, `ID` attributes with defaults, non-deterministic
    /// content models, and so on.
    #[must_use]
    pub fn declaration_errors(&self) -> &[ValidityError] {
        &self.declaration_errors
    }

    /// Records a validity error found while reading the DTD.
    pub fn push_declaration_error(&mut self, error: ValidityError) {
        self.declaration_errors.push(error);
    }

    /// Returns `true` when the DTD declares nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
            && self.attributes.is_empty()
            && self.general_entities.is_empty()
            && self.parameter_entities.is_empty()
            && self.notations.is_empty()
    }

    /// Checks the validity constraints that apply to attribute declarations,
    /// appending any violations to [`declaration_errors`](Self::declaration_errors).
    ///
    /// This is run for you by [`DtdParser`](crate::DtdParser) once the whole DTD
    /// has been read, because several of the constraints — for instance
    /// "a `NOTATION` attribute may not be declared for an `EMPTY` element" —
    /// depend on declarations that may come later.
    pub fn check_attribute_declarations(&mut self) {
        let mut errors = Vec::new();
        for list in self.attributes.values() {
            let element = list.element();
            let mut id_attributes = 0_usize;
            let mut notation_attributes = 0_usize;
            for declaration in list.iter() {
                match declaration.att_type() {
                    AttributeType::Id => {
                        id_attributes += 1;
                        if declaration.default().value().is_some() {
                            errors.push(ValidityError::detached(
                                ValidityErrorKind::IdAttributeHasDefault {
                                    element: element.to_owned(),
                                    attribute: declaration.name().to_owned(),
                                },
                            ));
                        }
                    }
                    AttributeType::Notation(names) => {
                        notation_attributes += 1;
                        if matches!(
                            self.elements.get(element).map(ElementDecl::content),
                            Some(ContentSpec::Empty)
                        ) {
                            errors.push(ValidityError::detached(
                                ValidityErrorKind::NotationOnEmptyElement {
                                    element: element.to_owned(),
                                },
                            ));
                        }
                        for notation in names {
                            if !self.notations.contains_key(notation) {
                                errors.push(ValidityError::detached(
                                    ValidityErrorKind::NotationNotDeclared {
                                        value: notation.clone(),
                                    },
                                ));
                            }
                        }
                    }
                    _ => {}
                }
                if let Some(duplicate) = declaration.att_type().first_duplicate_token() {
                    errors.push(ValidityError::detached(
                        ValidityErrorKind::DuplicateEnumerationToken {
                            element: element.to_owned(),
                            attribute: declaration.name().to_owned(),
                            token: duplicate.to_owned(),
                        },
                    ));
                }
            }
            if id_attributes > 1 {
                errors.push(ValidityError::detached(
                    ValidityErrorKind::MultipleIdAttributes {
                        element: element.to_owned(),
                    },
                ));
            }
            if notation_attributes > 1 {
                errors.push(ValidityError::detached(
                    ValidityErrorKind::MultipleNotationAttributes {
                        element: element.to_owned(),
                    },
                ));
            }
        }
        for entity in self.general_entities.values() {
            if let Some(notation) = entity.notation()
                && !self.notations.contains_key(notation)
            {
                errors.push(ValidityError::detached(
                    ValidityErrorKind::EntityNotationNotDeclared {
                        entity: entity.name().to_owned(),
                        notation: notation.to_owned(),
                    },
                ));
            }
        }
        self.declaration_errors.append(&mut errors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_element_declarations_keep_the_first_and_record_an_error() {
        let mut dtd = Dtd::new();
        assert!(dtd.add_element(ElementDecl::new("a", ContentSpec::Empty, false)));
        assert!(!dtd.add_element(ElementDecl::new("a", ContentSpec::Any, false)));
        assert_eq!(
            dtd.element("a").map(ElementDecl::content),
            Some(&ContentSpec::Empty)
        );
        assert!(matches!(
            dtd.declaration_errors().first().map(ValidityError::kind),
            Some(ValidityErrorKind::DuplicateElementDeclaration { .. })
        ));
    }

    #[test]
    fn duplicate_entity_declarations_keep_the_first_without_error() {
        let mut dtd = Dtd::new();
        assert!(dtd.add_entity(EntityDecl::new(
            "e",
            EntityKind::General,
            EntityDefinition::Internal("first".to_owned()),
            false,
        )));
        assert!(!dtd.add_entity(EntityDecl::new(
            "e",
            EntityKind::General,
            EntityDefinition::Internal("second".to_owned()),
            false,
        )));
        assert_eq!(
            dtd.general_entity("e").and_then(EntityDecl::internal_text),
            Some("first")
        );
        assert!(dtd.declaration_errors().is_empty());
    }

    #[test]
    fn attribute_checks_catch_two_id_attributes_and_an_id_default() {
        let mut dtd = Dtd::new();
        dtd.add_element(ElementDecl::new("a", ContentSpec::Any, false));
        dtd.add_attribute(
            "a",
            AttributeDecl::new("x", AttributeType::Id, DefaultDecl::Implied, false),
        );
        dtd.add_attribute(
            "a",
            AttributeDecl::new(
                "y",
                AttributeType::Id,
                DefaultDecl::Fixed("z".to_owned()),
                false,
            ),
        );
        dtd.check_attribute_declarations();
        let kinds: Vec<_> = dtd
            .declaration_errors()
            .iter()
            .map(ValidityError::kind)
            .collect();
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, ValidityErrorKind::IdAttributeHasDefault { .. }))
        );
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, ValidityErrorKind::MultipleIdAttributes { .. }))
        );
    }

    #[test]
    fn notation_attributes_on_empty_elements_are_rejected() {
        let mut dtd = Dtd::new();
        dtd.add_element(ElementDecl::new("a", ContentSpec::Empty, false));
        dtd.add_notation(NotationDecl::new("n", ExternalId::system("n"), false));
        dtd.add_attribute(
            "a",
            AttributeDecl::new(
                "x",
                AttributeType::Notation(vec!["n".to_owned()]),
                DefaultDecl::Implied,
                false,
            ),
        );
        dtd.check_attribute_declarations();
        assert!(dtd.declaration_errors().iter().any(|error| matches!(
            error.kind(),
            ValidityErrorKind::NotationOnEmptyElement { .. }
        )));
    }
}
