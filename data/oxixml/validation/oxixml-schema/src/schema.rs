//! The compiled schema: the component arenas plus the global symbol tables.

use std::collections::HashMap;

use oxixml_qname::ExpandedName;

use crate::components::{
    AttrGroupId, AttributeDeclaration, AttributeGroupDefinition, AttributeId, ComplexTypeInfo,
    Derivation, DerivationSet, ElementDeclaration, ElementId, GroupId, IdcId,
    IdentityConstraintDefinition, ModelGroupDefinition, NotationDeclaration, NotationId,
    SimpleTypeInfo, TypeDefinition, TypeId, Variety,
};
use crate::names::{XSD_NS, XsdVersion};

/// A compiled schema: every component, reachable by identifier or by name.
///
/// Build one with [`SchemaSet`](crate::SchemaSet); validate with
/// [`Schema::validate_document`](crate::Schema::validate_document) or
/// [`Schema::validate_str`](crate::Schema::validate_str).
#[derive(Debug, Clone)]
pub struct Schema {
    pub(crate) types: Vec<TypeDefinition>,
    pub(crate) elements: Vec<ElementDeclaration>,
    pub(crate) attributes: Vec<AttributeDeclaration>,
    pub(crate) groups: Vec<ModelGroupDefinition>,
    pub(crate) attribute_groups: Vec<AttributeGroupDefinition>,
    pub(crate) notations: Vec<NotationDeclaration>,
    pub(crate) identity_constraints: Vec<IdentityConstraintDefinition>,
    pub(crate) global_types: HashMap<ExpandedName, TypeId>,
    pub(crate) global_elements: HashMap<ExpandedName, ElementId>,
    pub(crate) global_attributes: HashMap<ExpandedName, AttributeId>,
    pub(crate) global_groups: HashMap<ExpandedName, GroupId>,
    pub(crate) global_attribute_groups: HashMap<ExpandedName, AttrGroupId>,
    pub(crate) global_notations: HashMap<ExpandedName, NotationId>,
    pub(crate) global_identity_constraints: HashMap<ExpandedName, IdcId>,
    pub(crate) any_type: TypeId,
    pub(crate) any_simple_type: TypeId,
    pub(crate) version: XsdVersion,
}

impl Schema {
    /// Which version of XML Schema this schema was assembled under.
    #[must_use]
    pub fn version(&self) -> XsdVersion {
        self.version
    }

    /// `xs:anyType`, the ur-type.
    #[must_use]
    pub fn any_type(&self) -> TypeId {
        self.any_type
    }

    /// `xs:anySimpleType`.
    #[must_use]
    pub fn any_simple_type(&self) -> TypeId {
        self.any_simple_type
    }

    /// The type definition behind an identifier.
    ///
    /// # Panics
    ///
    /// Panics when `id` did not come from **this** schema value. Every
    /// component identifier — [`TypeId`], [`ElementId`], … — indexes the arena
    /// of the [`Schema`] that handed it out and carries no record of which one
    /// that was, so an identifier kept across a recompilation, or taken from
    /// one schema and used against another, is meaningless here: it panics when
    /// the other arena is shorter and silently names an unrelated component
    /// when it is not. Identifiers are only meaningful for the lifetime of the
    /// [`Schema`] they were obtained from; nothing checks it for you.
    #[must_use]
    pub fn type_definition(&self, id: TypeId) -> &TypeDefinition {
        &self.types[id.index()]
    }

    /// The simple body of a type definition, when it has one.
    #[must_use]
    pub fn simple_type(&self, id: TypeId) -> Option<&SimpleTypeInfo> {
        self.type_definition(id).as_simple()
    }

    /// The complex body of a type definition, when it has one.
    #[must_use]
    pub fn complex_type(&self, id: TypeId) -> Option<&ComplexTypeInfo> {
        self.type_definition(id).as_complex()
    }

    /// The element declaration behind an identifier.
    ///
    /// # Panics
    ///
    /// Panics when `id` did not come from this schema value; see
    /// [`type_definition`](Self::type_definition).
    #[must_use]
    pub fn element_declaration(&self, id: ElementId) -> &ElementDeclaration {
        &self.elements[id.index()]
    }

    /// The attribute declaration behind an identifier.
    ///
    /// # Panics
    ///
    /// Panics when `id` did not come from this schema value; see
    /// [`type_definition`](Self::type_definition).
    #[must_use]
    pub fn attribute_declaration(&self, id: AttributeId) -> &AttributeDeclaration {
        &self.attributes[id.index()]
    }

    /// The notation declaration behind an identifier.
    ///
    /// # Panics
    ///
    /// Panics when `id` did not come from this schema value; see
    /// [`type_definition`](Self::type_definition).
    #[must_use]
    pub fn notation_declaration(&self, id: NotationId) -> &NotationDeclaration {
        &self.notations[id.index()]
    }

    /// The identity constraint behind an identifier.
    ///
    /// # Panics
    ///
    /// Panics when `id` did not come from this schema value; see
    /// [`type_definition`](Self::type_definition).
    #[must_use]
    pub fn identity_constraint(&self, id: IdcId) -> &IdentityConstraintDefinition {
        &self.identity_constraints[id.index()]
    }

    /// Looks a global type definition up by name.
    #[must_use]
    pub fn find_type(&self, namespace: Option<&str>, local: &str) -> Option<TypeId> {
        self.global_types
            .get(&ExpandedName::new(namespace, local))
            .copied()
    }

    /// Looks a global element declaration up by name.
    #[must_use]
    pub fn find_element(&self, namespace: Option<&str>, local: &str) -> Option<ElementId> {
        self.global_elements
            .get(&ExpandedName::new(namespace, local))
            .copied()
    }

    /// Looks a global attribute declaration up by name.
    #[must_use]
    pub fn find_attribute(&self, namespace: Option<&str>, local: &str) -> Option<AttributeId> {
        self.global_attributes
            .get(&ExpandedName::new(namespace, local))
            .copied()
    }

    /// Looks a notation declaration up by name.
    #[must_use]
    pub fn find_notation(&self, namespace: Option<&str>, local: &str) -> Option<NotationId> {
        self.global_notations
            .get(&ExpandedName::new(namespace, local))
            .copied()
    }

    /// A built-in datatype, by its local name in the XML Schema namespace.
    #[must_use]
    pub fn builtin(&self, local: &str) -> Option<TypeId> {
        self.find_type(Some(XSD_NS), local)
    }

    /// Every global element declaration, in an unspecified order.
    pub fn global_elements(&self) -> impl Iterator<Item = (&ExpandedName, ElementId)> {
        self.global_elements.iter().map(|(name, id)| (name, *id))
    }

    /// Every global type definition, in an unspecified order.
    pub fn global_types(&self) -> impl Iterator<Item = (&ExpandedName, TypeId)> {
        self.global_types.iter().map(|(name, id)| (name, *id))
    }

    /// The number of type definitions, including the built-ins.
    #[must_use]
    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    /// Whether `candidate` is `base`, or derives from it by any number of steps.
    #[must_use]
    pub fn is_derived_from(&self, candidate: TypeId, base: TypeId) -> bool {
        self.is_derived_from_bounded(candidate, base, self.types.len())
    }

    /// [`is_derived_from`](Self::is_derived_from), with `budget` bounding the
    /// recursion through nested unions so a pathological (and, in a compiled
    /// schema, unreachable) member cycle cannot recurse unboundedly.
    fn is_derived_from_bounded(&self, candidate: TypeId, base: TypeId, budget: usize) -> bool {
        if candidate == base {
            return true;
        }
        // Part 1 §3.16.6.3 *Type Derivation OK (Simple)*, clause 2.2.4: when
        // `base`'s {variety} is union, `candidate` is validly derived from it
        // if "2.2.4.2 D is validly ·derived· from a type definition M in B's
        // ·transitive membership·" — this is what lets `xsi:type` name a union
        // member directly, rather than only the union itself
        // (`union/s3_16_2v05`, `s3_16_2v07`) — but only when "2.2.4.3 the
        // {facets} property of B **and of any intervening union datatypes** is
        // empty", since otherwise naming a member would slip past the facets
        // the union itself imposes (`saxonData/Simple/simple014` …
        // `simple016`). The recursion re-applies the same test at every
        // nested union, which is what "intervening" asks for.
        if budget > 0
            && let Some(members) = self.union_member_types(base)
            && self
                .simple_type(base)
                .is_some_and(|simple| simple.facets.is_unconstrained())
            && members
                .iter()
                .any(|member| self.is_derived_from_bounded(candidate, *member, budget - 1))
        {
            return true;
        }
        let mut current = candidate;
        let mut guard = 0;
        loop {
            if current == base {
                return true;
            }
            let Some(next) = self.type_definition(current).base else {
                return false;
            };
            if next == current {
                return false;
            }
            current = next;
            guard += 1;
            if guard > self.types.len() {
                return false;
            }
        }
    }

    /// The set of derivation methods used on the path from `candidate` up to
    /// `base`, or [`None`] when `candidate` does not derive from `base`.
    #[must_use]
    pub fn derivation_path(&self, candidate: TypeId, base: TypeId) -> Option<DerivationSet> {
        let mut used = DerivationSet::empty();
        let mut current = candidate;
        let mut guard = 0;
        loop {
            if current == base {
                return Some(used);
            }
            let definition = self.type_definition(current);
            match definition.derivation {
                Derivation::Extension => used.extension = true,
                Derivation::Restriction => used.restriction = true,
                Derivation::List => used.list = true,
                Derivation::Union => used.union = true,
                Derivation::Substitution => used.substitution = true,
            }
            let next = definition.base?;
            if next == current {
                return None;
            }
            current = next;
            guard += 1;
            if guard > self.types.len() {
                return None;
            }
        }
    }

    /// Whether `candidate` is `head`, or is in `head`'s substitution group.
    ///
    /// This is the *whole* `{substitution group}` of §3.3.6.3 — the transitive
    /// closure of `substitutionGroup` — not the ·actual· substitution group
    /// that decides what a particle referring to `head` may match. Under XML
    /// Schema 1.0 the two differ: an `abstract` member is in the first and not
    /// the second, since no element information item is ever locally valid
    /// against it.
    #[must_use]
    pub fn is_substitutable(&self, candidate: ElementId, head: ElementId) -> bool {
        candidate == head
            || self
                .element_declaration(head)
                .substitutable_by
                .contains(&candidate)
    }

    /// Whether a member of a substitution group belongs to the *actual*
    /// substitution group of its head — the set that decides what a particle
    /// referring to the head can match, and what can compete with it.
    ///
    /// XML Schema **1.0** leaves an `abstract` declaration out: no element
    /// information item is ever locally valid against one, so a particle
    /// referring to the head can never match it. XML Schema 1.1 deliberately
    /// puts it back — appendix G.1.9, "Abstract elements may now appear in
    /// substitution groups", adopted to resolve [W3C bug 4337][bug] — so that
    /// the Unique Particle Attribution and Element Declarations Consistent
    /// constraints take abstract members into account. The change is about
    /// those constraints only: an abstract declaration still cannot be
    /// instantiated under either version.
    ///
    /// [bug]: https://www.w3.org/Bugs/Public/show_bug.cgi?id=4337
    pub(crate) fn is_actual_substitution_member(&self, member: ElementId) -> bool {
        self.version.is_1_1() || !self.element_declaration(member).abstract_
    }

    /// The primitive ancestor of a simple type, when it is atomic.
    #[must_use]
    pub fn primitive_of(&self, id: TypeId) -> Option<crate::value::Primitive> {
        self.simple_type(id).and_then(|simple| simple.primitive)
    }

    /// Whether the type is a list, and if so its item type.
    #[must_use]
    pub fn list_item_type(&self, id: TypeId) -> Option<TypeId> {
        match &self.simple_type(id)?.variety {
            Variety::List(item) => Some(*item),
            _ => None,
        }
    }

    /// The member types of a union simple type.
    #[must_use]
    pub fn union_member_types(&self, id: TypeId) -> Option<Vec<TypeId>> {
        match &self.simple_type(id)?.variety {
            Variety::Union(members) => Some(members.clone()),
            _ => None,
        }
    }

    /// A short human-readable name for a type, used in diagnostics.
    #[must_use]
    pub fn type_name(&self, id: TypeId) -> String {
        self.type_definition(id).display_name()
    }
}
