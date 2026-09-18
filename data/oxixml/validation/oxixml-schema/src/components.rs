//! The schema component model of
//! [XML Schema Part 1 §3](https://www.w3.org/TR/xmlschema11-1/#components).
//!
//! Components live in arenas owned by the [`Schema`](crate::Schema) and are
//! referred to by small `Copy` identifiers ([`TypeId`], [`ElementId`], …). That
//! keeps the graph — which is deeply cyclic, since a type may contain an element
//! declaration whose type is the type itself — free of reference counting and of
//! `unsafe`, and makes every component cheap to pass around.

use std::collections::BTreeSet;

use oxixml_qname::ExpandedName;
use oxixml_regex::Regex;

use crate::automaton::ContentModel;
use crate::value::{ExplicitTimezone, NamespaceContext, Primitive, Value, WhiteSpace};

macro_rules! component_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub(crate) u32);

        impl $name {
            /// The index of the component in its arena.
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

component_id!(
    /// Identifies a simple or complex type definition.
    TypeId
);
component_id!(
    /// Identifies an element declaration.
    ElementId
);
component_id!(
    /// Identifies an attribute declaration.
    AttributeId
);
component_id!(
    /// Identifies a named model group definition.
    GroupId
);
component_id!(
    /// Identifies an attribute group definition.
    AttrGroupId
);
component_id!(
    /// Identifies a notation declaration.
    NotationId
);
component_id!(
    /// Identifies an identity-constraint definition.
    IdcId
);

/// A set of derivation methods, as written in `block`, `final`, `blockDefault`
/// and `finalDefault`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DerivationSet {
    /// `extension` is blocked/final.
    pub extension: bool,
    /// `restriction` is blocked/final.
    pub restriction: bool,
    /// `list` is final (simple types only).
    pub list: bool,
    /// `union` is final (simple types only).
    pub union: bool,
    /// `substitution` is blocked (element declarations only).
    pub substitution: bool,
}

impl DerivationSet {
    /// The empty set.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            extension: false,
            restriction: false,
            list: false,
            union: false,
            substitution: false,
        }
    }

    /// Every method (`#all`).
    #[must_use]
    pub const fn all() -> Self {
        Self {
            extension: true,
            restriction: true,
            list: true,
            union: true,
            substitution: true,
        }
    }

    /// Whether the set contains `method`.
    #[must_use]
    pub const fn contains(self, method: Derivation) -> bool {
        match method {
            Derivation::Extension => self.extension,
            Derivation::Restriction => self.restriction,
            Derivation::List => self.list,
            Derivation::Union => self.union,
            Derivation::Substitution => self.substitution,
        }
    }

    /// The union of two sets.
    #[must_use]
    pub const fn union_with(self, other: Self) -> Self {
        Self {
            extension: self.extension || other.extension,
            restriction: self.restriction || other.restriction,
            list: self.list || other.list,
            union: self.union || other.union,
            substitution: self.substitution || other.substitution,
        }
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        !(self.extension || self.restriction || self.list || self.union || self.substitution)
    }
}

/// One derivation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Derivation {
    /// Derivation by extension.
    Extension,
    /// Derivation by restriction.
    Restriction,
    /// Construction of a list type.
    List,
    /// Construction of a union type.
    Union,
    /// Substitution (element declarations only).
    Substitution,
}

impl std::fmt::Display for Derivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Extension => "extension",
            Self::Restriction => "restriction",
            Self::List => "list",
            Self::Union => "union",
            Self::Substitution => "substitution",
        })
    }
}

/// A `default` or `fixed` value on an element or attribute declaration.
#[derive(Debug, Clone)]
pub struct ValueConstraint {
    /// `true` for `fixed`, `false` for `default`.
    pub fixed: bool,
    /// The lexical form exactly as written in the schema document.
    pub lexical: String,
    /// The namespace bindings in force where it was written (a `QName`-typed
    /// default has to be read in them).
    pub namespaces: NamespaceContext,
    /// The value it denotes, once the type definition is known.
    pub value: Option<Value>,
    /// The `whiteSpace`-normalized lexical form.
    pub normalized: String,
}

impl ValueConstraint {
    /// Builds a constraint from its lexical form.
    #[must_use]
    pub fn new(fixed: bool, lexical: impl Into<String>, namespaces: NamespaceContext) -> Self {
        let lexical = lexical.into();
        Self {
            fixed,
            normalized: lexical.clone(),
            lexical,
            namespaces,
            value: None,
        }
    }
}

/// The `processContents` of a wildcard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessContents {
    /// The matched element must be declared, and is fully assessed.
    #[default]
    Strict,
    /// The matched element is assessed if a declaration can be found.
    Lax,
    /// The matched element and its descendants are not assessed at all.
    Skip,
}

impl ProcessContents {
    /// Parses the attribute value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "strict" => Some(Self::Strict),
            "lax" => Some(Self::Lax),
            "skip" => Some(Self::Skip),
            _ => None,
        }
    }
}

/// The *namespace constraint* of a wildcard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceConstraint {
    /// `##any`.
    Any,
    /// `##other`, or a 1.1 `notNamespace`: any namespace **not** in the set.
    Not(BTreeSet<Option<String>>),
    /// An explicit list; [`None`] inside stands for `##local` (absent).
    Enumeration(BTreeSet<Option<String>>),
}

impl NamespaceConstraint {
    /// Whether `namespace` (absent as [`None`]) is allowed.
    #[must_use]
    pub fn allows(&self, namespace: Option<&str>) -> bool {
        let key = namespace.map(str::to_owned);
        match self {
            Self::Any => true,
            Self::Not(set) => !set.contains(&key),
            Self::Enumeration(set) => set.contains(&key),
        }
    }

    /// Whether this constraint allows every namespace the `other` one allows.
    #[must_use]
    pub fn is_superset_of(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Any, _) => true,
            (_, Self::Any) => false,
            (Self::Enumeration(a), Self::Enumeration(b)) => b.is_subset(a),
            (Self::Not(a), Self::Not(b)) => a.is_subset(b),
            (Self::Not(a), Self::Enumeration(b)) => a.is_disjoint(b),
            (Self::Enumeration(_), Self::Not(_)) => false,
        }
    }

    /// Whether some namespace is allowed by both constraints.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Any, _) | (_, Self::Any) => true,
            (Self::Enumeration(a), Self::Enumeration(b)) => !a.is_disjoint(b),
            (Self::Not(_), Self::Not(_)) => true,
            (Self::Not(a), Self::Enumeration(b)) | (Self::Enumeration(b), Self::Not(a)) => {
                b.iter().any(|item| !a.contains(item))
            }
        }
    }

    /// The intersection, when it is expressible (XSD 1.0 leaves a few cases
    /// undefined; those return [`None`]).
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        Some(match (self, other) {
            (Self::Any, rest) | (rest, Self::Any) => rest.clone(),
            (Self::Enumeration(a), Self::Enumeration(b)) => {
                Self::Enumeration(a.intersection(b).cloned().collect())
            }
            (Self::Not(a), Self::Enumeration(b)) | (Self::Enumeration(b), Self::Not(a)) => {
                Self::Enumeration(
                    b.iter()
                        .filter(|item| !a.contains(*item))
                        .cloned()
                        .collect(),
                )
            }
            (Self::Not(a), Self::Not(b)) => {
                let merged: BTreeSet<Option<String>> = a.union(b).cloned().collect();
                Self::Not(merged)
            }
        })
    }

    /// The union, when it is expressible.
    #[must_use]
    pub fn union_with(&self, other: &Self) -> Option<Self> {
        Some(match (self, other) {
            (Self::Any, _) | (_, Self::Any) => Self::Any,
            (Self::Enumeration(a), Self::Enumeration(b)) => {
                Self::Enumeration(a.union(b).cloned().collect())
            }
            (Self::Not(a), Self::Enumeration(b)) | (Self::Enumeration(b), Self::Not(a)) => {
                let remaining: BTreeSet<Option<String>> = a
                    .iter()
                    .filter(|item| !b.contains(*item))
                    .cloned()
                    .collect();
                if remaining.is_empty() {
                    Self::Any
                } else {
                    Self::Not(remaining)
                }
            }
            (Self::Not(a), Self::Not(b)) => {
                let shared: BTreeSet<Option<String>> = a.intersection(b).cloned().collect();
                if shared.is_empty() {
                    Self::Any
                } else {
                    Self::Not(shared)
                }
            }
        })
    }
}

/// An element or attribute wildcard (`xs:any` / `xs:anyAttribute`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wildcard {
    /// Which namespaces it matches.
    pub namespace: NamespaceConstraint,
    /// How matched content is assessed.
    pub process_contents: ProcessContents,
    /// The XSD 1.1 {disallowed names} of `notQName`.
    pub disallowed: DisallowedNames,
}

/// The XSD 1.1 *{disallowed names}* of a wildcard — what `notQName` writes
/// (Part 1 §3.10.1).
///
/// A wildcard whose namespace constraint admits a name still does not match it
/// when the name is one of [`DisallowedNames::names`], or when `##defined` is
/// in force and the name is that of a *global* declaration of the same kind.
/// `##definedSibling` — which an *element* wildcard alone may write — is
/// resolved during compilation into [`DisallowedNames::names`], because the
/// declarations it names are exactly those of the content model the wildcard
/// sits in and are all known by then.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DisallowedNames {
    /// The QNames written out, in the order they were written.
    pub names: Vec<ExpandedName>,
    /// `##defined`.
    pub defined: bool,
    /// `##definedSibling`, until compilation resolves it.
    pub defined_sibling: bool,
}

impl DisallowedNames {
    /// Whether nothing is disallowed, which is the 1.0 case and the common one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty() && !self.defined && !self.defined_sibling
    }

    /// The union: what a wildcard formed by *intersecting* two wildcards
    /// disallows (§3.10.6, attribute-wildcard intersection).
    #[must_use]
    pub fn union_with(&self, other: &Self) -> Self {
        let mut names = self.names.clone();
        for name in &other.names {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        Self {
            names,
            defined: self.defined || other.defined,
            defined_sibling: self.defined_sibling || other.defined_sibling,
        }
    }

    /// The intersection: what a wildcard formed by *uniting* two wildcards
    /// disallows.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            names: self
                .names
                .iter()
                .filter(|name| other.names.contains(name))
                .cloned()
                .collect(),
            defined: self.defined && other.defined,
            defined_sibling: self.defined_sibling && other.defined_sibling,
        }
    }
}

/// The {disallowed names} of the *wildcard union* of two wildcards
/// (§3.10.6.3, *Attribute Wildcard Union*).
///
/// The union disallows "all and only … 1 QName members of `O1 . {disallowed
/// names}` that are **not allowed by** `O2`, as defined in *Wildcard allows
/// Expanded Name* (§3.10.4.2); 2 QName members of `O2 . {disallowed names}`
/// that are not allowed by `O1`; 3 the keyword `defined` if it is contained in
/// **both**". Being "not allowed by" the other wildcard covers a name the other
/// wildcard's *namespace constraint* excludes as well as one it disallows by
/// name, which is why this cannot be a plain set intersection of the two name
/// lists (`saxonData/Wild/wild046`: `xml:lang` is disallowed by one wildcard
/// and out of the `##local` namespace of the other, so the union disallows it).
#[must_use]
pub fn union_disallowed_names(left: &Wildcard, right: &Wildcard) -> DisallowedNames {
    let mut names = Vec::new();
    for (mine, theirs) in [(left, right), (right, left)] {
        for name in &mine.disallowed.names {
            if theirs.matches(name.namespace(), name.local_name(), false) {
                continue;
            }
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    DisallowedNames {
        names,
        defined: left.disallowed.defined && right.disallowed.defined,
        defined_sibling: left.disallowed.defined_sibling && right.disallowed.defined_sibling,
    }
}

impl Wildcard {
    /// A wildcard matching everything, laxly.
    #[must_use]
    pub fn any_lax() -> Self {
        Self {
            namespace: NamespaceConstraint::Any,
            process_contents: ProcessContents::Lax,
            disallowed: DisallowedNames::default(),
        }
    }

    /// Whether the wildcard matches `namespace`:`local`.
    ///
    /// `globally_defined` answers "is there a global declaration of this kind
    /// with this name?", which only `##defined` needs; a caller with no schema
    /// in hand passes `false`.
    #[must_use]
    pub fn matches(&self, namespace: Option<&str>, local: &str, globally_defined: bool) -> bool {
        if !self.namespace.allows(namespace) {
            return false;
        }
        if self.disallowed.is_empty() {
            return true;
        }
        if self.disallowed.defined && globally_defined {
            return false;
        }
        !self
            .disallowed
            .names
            .contains(&ExpandedName::new(namespace, local))
    }
}

/// `maxOccurs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxOccurs {
    /// A finite bound.
    Bounded(u32),
    /// `unbounded`.
    Unbounded,
}

impl MaxOccurs {
    /// Whether `count` is within the bound.
    #[must_use]
    pub const fn allows(self, count: u32) -> bool {
        match self {
            Self::Bounded(max) => count <= max,
            Self::Unbounded => true,
        }
    }

    /// Whether the bound is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        matches!(self, Self::Bounded(0))
    }

    /// The bound as a number, or [`None`] for `unbounded`.
    #[must_use]
    pub const fn bound(self) -> Option<u32> {
        match self {
            Self::Bounded(max) => Some(max),
            Self::Unbounded => None,
        }
    }
}

impl std::fmt::Display for MaxOccurs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bounded(max) => write!(f, "{max}"),
            Self::Unbounded => f.write_str("unbounded"),
        }
    }
}

/// The compositor of a model group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compositor {
    /// `xs:sequence`.
    Sequence,
    /// `xs:choice`.
    Choice,
    /// `xs:all`.
    All,
}

/// A model group: a compositor and its particles.
#[derive(Debug, Clone)]
pub struct ModelGroup {
    /// `sequence`, `choice` or `all`.
    pub compositor: Compositor,
    /// The particles, in order.
    pub particles: Vec<Particle>,
}

/// The *term* of a particle.
#[derive(Debug, Clone)]
pub enum Term {
    /// An element declaration.
    Element(ElementId),
    /// A wildcard.
    Wildcard(Wildcard),
    /// A nested model group (named group references are inlined at compile
    /// time).
    Group(ModelGroup),
}

/// A particle: a term with occurrence bounds.
#[derive(Debug, Clone)]
pub struct Particle {
    /// `minOccurs`.
    pub min_occurs: u32,
    /// `maxOccurs`.
    pub max_occurs: MaxOccurs,
    /// The term.
    pub term: Term,
}

impl Particle {
    /// Whether the particle can match the empty sequence.
    #[must_use]
    pub fn is_emptiable(&self) -> bool {
        if self.min_occurs == 0 {
            return true;
        }
        match &self.term {
            Term::Element(_) | Term::Wildcard(_) => false,
            Term::Group(group) => match group.compositor {
                Compositor::Choice => {
                    group.particles.is_empty() || group.particles.iter().any(Self::is_emptiable)
                }
                Compositor::Sequence | Compositor::All => {
                    group.particles.iter().all(Self::is_emptiable)
                }
            },
        }
    }
}

/// A named model group definition (`xs:group` at the top level).
#[derive(Debug, Clone)]
pub struct ModelGroupDefinition {
    /// Its name.
    pub name: ExpandedName,
    /// The model group it defines.
    pub model_group: Option<ModelGroup>,
}

/// How an attribute may or must appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeUseKind {
    /// `use="optional"`.
    Optional,
    /// `use="required"`.
    Required,
    /// `use="prohibited"`.
    Prohibited,
}

/// An *attribute use*: an attribute declaration as used by a complex type.
#[derive(Debug, Clone)]
pub struct AttributeUse {
    /// The declaration used.
    pub attribute: AttributeId,
    /// Optional, required or prohibited.
    pub kind: AttributeUseKind,
    /// A `default`/`fixed` value overriding the declaration's own.
    pub value_constraint: Option<ValueConstraint>,
    /// Whether the use was written with `inheritable="true"` (1.1).
    pub inheritable: bool,
}

/// An attribute group definition (`xs:attributeGroup` at the top level).
#[derive(Debug, Clone, Default)]
pub struct AttributeGroupDefinition {
    /// Its name.
    pub name: Option<ExpandedName>,
    /// The attribute uses it contributes.
    pub attribute_uses: Vec<AttributeUse>,
    /// The attribute wildcard it contributes.
    pub attribute_wildcard: Option<Wildcard>,
}

/// A `xs:notation` declaration.
#[derive(Debug, Clone)]
pub struct NotationDeclaration {
    /// Its name.
    pub name: ExpandedName,
    /// The `public` identifier.
    pub public_id: Option<String>,
    /// The `system` identifier.
    pub system_id: Option<String>,
}

/// The category of an identity constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdcCategory {
    /// `xs:key`.
    Key,
    /// `xs:keyref`.
    KeyRef,
    /// `xs:unique`.
    Unique,
}

impl std::fmt::Display for IdcCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Key => "key",
            Self::KeyRef => "keyref",
            Self::Unique => "unique",
        })
    }
}

/// An identity-constraint definition (`xs:key`, `xs:keyref` or `xs:unique`).
#[derive(Debug, Clone)]
pub struct IdentityConstraintDefinition {
    /// Its name.
    pub name: ExpandedName,
    /// Key, key reference or uniqueness.
    pub category: IdcCategory,
    /// The `selector` XPath.
    pub selector: crate::xpath::PathSet,
    /// The `field` XPaths.
    pub fields: Vec<crate::xpath::PathSet>,
    /// For a `keyref`, the constraint it refers to.
    pub refer: Option<IdcId>,
    /// For a `keyref`, the name it referred to before resolution.
    pub refer_name: Option<ExpandedName>,
}

/// An element declaration.
#[derive(Debug, Clone)]
pub struct ElementDeclaration {
    /// Its expanded name.
    pub name: ExpandedName,
    /// Its type definition.
    pub type_id: TypeId,
    /// Whether it is a *global* (top-level) declaration.
    pub global: bool,
    /// `nillable`.
    pub nillable: bool,
    /// `abstract`.
    pub abstract_: bool,
    /// The `default`/`fixed` value.
    pub value_constraint: Option<ValueConstraint>,
    /// Its identity constraints.
    pub identity_constraints: Vec<IdcId>,
    /// The heads of its substitution group (1.1 allows more than one).
    pub substitution_group: Vec<ElementId>,
    /// `block` (already defaulted from `blockDefault`).
    pub block: DerivationSet,
    /// `final` (already defaulted from `finalDefault`).
    pub final_: DerivationSet,
    /// The members of the *substitution group* this declaration heads,
    /// transitively. Filled in during compilation.
    pub substitutable_by: Vec<ElementId>,
    /// The {type table} of XSD 1.1 conditional type assignment: the
    /// `xs:alternative` children, in order. Empty for a declaration that has
    /// none, which is every 1.0 declaration.
    pub alternatives: Vec<TypeAlternative>,
}

/// An attribute declaration.
#[derive(Debug, Clone)]
pub struct AttributeDeclaration {
    /// Its expanded name.
    pub name: ExpandedName,
    /// Its simple type definition.
    pub type_id: TypeId,
    /// Whether it is a *global* (top-level) declaration.
    pub global: bool,
    /// The `default`/`fixed` value.
    pub value_constraint: Option<ValueConstraint>,
    /// Whether the declaration is inheritable (1.1).
    pub inheritable: bool,
}

/// The *variety* of a simple type definition.
#[derive(Debug, Clone)]
pub enum Variety {
    /// `xs:anySimpleType` itself: no variety.
    Absent,
    /// An atomic type, with its primitive ancestor.
    Atomic(Primitive),
    /// A list type, with its item type.
    List(TypeId),
    /// A union type, with its member types.
    Union(Vec<TypeId>),
}

/// The constraining facets that apply to a simple type definition, merged along
/// the whole restriction chain.
#[derive(Debug, Clone, Default)]
pub struct Facets {
    /// `length`.
    pub length: Option<usize>,
    /// `minLength`.
    pub min_length: Option<usize>,
    /// `maxLength`.
    pub max_length: Option<usize>,
    /// `pattern`: one entry per restriction step, each a disjunction.
    pub patterns: Vec<Vec<Regex>>,
    /// The lexical forms of the patterns, in the same shape, for diagnostics.
    pub pattern_sources: Vec<Vec<String>>,
    /// `enumeration`, as values; [`None`] means "not constrained".
    pub enumeration: Option<Vec<Value>>,
    /// The lexical forms of the enumeration members.
    pub enumeration_lexical: Vec<String>,
    /// `whiteSpace`.
    pub white_space: WhiteSpace,
    /// Whether `whiteSpace` was fixed by a `fixed="true"` facet or by the
    /// primitive itself.
    pub white_space_fixed: bool,
    /// The names of the constraining facets carrying `fixed="true"` anywhere
    /// along the restriction chain that produced these facets.
    ///
    /// Datatypes §4.1.3 / Structures §3.16.6.4 (*Simple Type Restriction
    /// (Facets)*): a facet fixed by a base type may be repeated with the same
    /// value by a restriction, but never given a different one. `pattern`,
    /// `enumeration` and `assertion` never appear here — they take no `fixed`
    /// attribute.
    pub fixed: std::collections::BTreeSet<String>,
    /// `minInclusive`.
    pub min_inclusive: Option<Value>,
    /// `maxInclusive`.
    pub max_inclusive: Option<Value>,
    /// `minExclusive`.
    pub min_exclusive: Option<Value>,
    /// `maxExclusive`.
    pub max_exclusive: Option<Value>,
    /// `totalDigits`.
    pub total_digits: Option<usize>,
    /// `fractionDigits`.
    pub fraction_digits: Option<usize>,
    /// `explicitTimezone` (1.1).
    pub explicit_timezone: Option<ExplicitTimezone>,
    /// `assertion` (1.1); stored but only evaluated for the documented subset.
    pub assertions: Vec<Assertion>,
}

impl Facets {
    /// Whether these facets constrain nothing.
    ///
    /// `whiteSpace` is deliberately ignored: every simple type has one, so
    /// counting it would make the "{facets} … is empty" test of *Type
    /// Derivation OK (Simple)* clause 2.2.4.3 unsatisfiable.
    #[must_use]
    pub fn is_unconstrained(&self) -> bool {
        self.length.is_none()
            && self.min_length.is_none()
            && self.max_length.is_none()
            && self.patterns.is_empty()
            && self.enumeration.is_none()
            && self.min_inclusive.is_none()
            && self.max_inclusive.is_none()
            && self.min_exclusive.is_none()
            && self.max_exclusive.is_none()
            && self.total_digits.is_none()
            && self.fraction_digits.is_none()
            && self.explicit_timezone.is_none()
            && self.assertions.is_empty()
    }
}

/// An XSD 1.1 assertion (`xs:assert` or the `assertion` facet).
#[derive(Debug, Clone)]
pub struct Assertion {
    /// The `test` XPath expression, verbatim.
    pub test: String,
    /// The namespace bindings in force where it was written.
    pub namespaces: NamespaceContext,
    /// `xpathDefaultNamespace` in force where it was written, already resolved
    /// from `##targetNamespace`/`##defaultNamespace`/`##local` to the namespace
    /// name it denotes (§3.13.1).
    pub xpath_default_namespace: Option<String>,
}

/// An XSD 1.1 *type alternative* — one `xs:alternative` of an element
/// declaration's {type table} (Part 1 §3.3.2, conditional type assignment).
#[derive(Debug, Clone)]
pub struct TypeAlternative {
    /// The `test` XPath expression; [`None`] for the *default* alternative,
    /// which is the one written without a `test`.
    pub test: Option<Assertion>,
    /// The type this alternative selects.
    pub type_id: TypeId,
}

/// A simple type definition.
#[derive(Debug, Clone)]
pub struct SimpleTypeInfo {
    /// Atomic, list or union.
    pub variety: Variety,
    /// The primitive ancestor, when the variety is atomic.
    pub primitive: Option<Primitive>,
    /// The merged constraining facets.
    pub facets: Facets,
}

/// The *content type* of a complex type definition.
#[derive(Debug, Clone)]
pub enum ContentType {
    /// No content is allowed.
    Empty,
    /// Character content validated against a simple type.
    Simple(TypeId),
    /// Element-only content.
    ElementOnly(Box<Particle>),
    /// Mixed content.
    Mixed(Box<Particle>),
}

impl ContentType {
    /// The particle, for the two element-bearing varieties.
    #[must_use]
    pub fn particle(&self) -> Option<&Particle> {
        match self {
            Self::ElementOnly(particle) | Self::Mixed(particle) => Some(particle),
            _ => None,
        }
    }

    /// Whether character data other than whitespace is allowed.
    #[must_use]
    pub fn allows_character_data(&self) -> bool {
        matches!(self, Self::Simple(_) | Self::Mixed(_))
    }
}

/// A complex type definition.
#[derive(Debug, Clone)]
pub struct ComplexTypeInfo {
    /// Whether it derives by extension or by restriction.
    pub derivation: Derivation,
    /// `abstract`.
    pub abstract_: bool,
    /// Its attribute uses.
    pub attribute_uses: Vec<AttributeUse>,
    /// Its attribute wildcard.
    pub attribute_wildcard: Option<Wildcard>,
    /// Its content type.
    pub content: ContentType,
    /// The compiled content-model automaton.
    pub content_model: Option<ContentModel>,
    /// 1.1 assertions.
    pub assertions: Vec<Assertion>,
    /// 1.1 `openContent`, as the wildcard it contributes and its mode.
    pub open_content: Option<OpenContent>,
}

/// XSD 1.1 `xs:openContent`.
#[derive(Debug, Clone)]
pub struct OpenContent {
    /// `interleave` (the default) or `suffix`.
    pub interleave: bool,
    /// The wildcard; absent for `mode="none"`.
    pub wildcard: Option<Wildcard>,
}

/// The two kinds of type definition.
#[derive(Debug, Clone)]
pub enum TypeKind {
    /// A simple type definition.
    Simple(Box<SimpleTypeInfo>),
    /// A complex type definition.
    Complex(Box<ComplexTypeInfo>),
}

/// A type definition — simple or complex — as stored in the schema arena.
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    /// Its name, when it is a top-level (named) definition.
    pub name: Option<ExpandedName>,
    /// Its base type definition; [`None`] only for `xs:anyType`.
    pub base: Option<TypeId>,
    /// How it derives from its base.
    pub derivation: Derivation,
    /// `final`.
    pub final_: DerivationSet,
    /// `block` (complex types only).
    pub block: DerivationSet,
    /// The simple or complex body.
    pub kind: TypeKind,
}

impl TypeDefinition {
    /// The simple body, when this is a simple type definition.
    #[must_use]
    pub fn as_simple(&self) -> Option<&SimpleTypeInfo> {
        match &self.kind {
            TypeKind::Simple(simple) => Some(simple),
            TypeKind::Complex(_) => None,
        }
    }

    /// The complex body, when this is a complex type definition.
    #[must_use]
    pub fn as_complex(&self) -> Option<&ComplexTypeInfo> {
        match &self.kind {
            TypeKind::Complex(complex) => Some(complex),
            TypeKind::Simple(_) => None,
        }
    }

    /// Whether this is a simple type definition.
    #[must_use]
    pub fn is_simple(&self) -> bool {
        matches!(self.kind, TypeKind::Simple(_))
    }

    /// A displayable name (`{ns}local`, or `#anonymous`).
    #[must_use]
    pub fn display_name(&self) -> String {
        match &self.name {
            Some(name) => name.to_string(),
            None => "#anonymous".to_owned(),
        }
    }
}
