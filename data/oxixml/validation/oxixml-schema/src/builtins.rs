//! The built-in datatype hierarchy of
//! [XML Schema Part 2](https://www.w3.org/TR/xmlschema11-2/), constructed as
//! ordinary schema components.
//!
//! Nothing here is special-cased at validation time: `xs:byte` really is a
//! restriction of `xs:short` with `minInclusive`/`maxInclusive` facets, and
//! `xs:NCName` really is a restriction of `xs:Name` with the pattern
//! `[\i-[:]][\c-[:]]*`, so the generic facet machinery in
//! [`crate::simple`] does all the work.

use std::cell::RefCell;
use std::collections::HashMap;

use oxixml_qname::ExpandedName;

use crate::automaton::ContentModel;
use crate::components::Compositor;
use crate::components::{
    AttrGroupId, AttributeDeclaration, AttributeGroupDefinition, AttributeId, AttributeUse,
    AttributeUseKind, ComplexTypeInfo, ContentType, Derivation, DerivationSet, Facets, MaxOccurs,
    ModelGroup, NamespaceConstraint, Particle, ProcessContents, SimpleTypeInfo, Term,
    TypeDefinition, TypeId, TypeKind, Variety, Wildcard,
};
use crate::error::SchemaError;
use crate::names::{XML_NS, XSD_NS, XSI_NS, XsdVersion};
use crate::schema::Schema;
use crate::simple::{RawFacet, apply_facets};
use crate::value::{NamespaceContext, Primitive, WhiteSpace};

/// The nineteen primitive datatypes, in the order Part 2 lists them.
const PRIMITIVES: &[Primitive] = &[
    Primitive::String,
    Primitive::Boolean,
    Primitive::Decimal,
    Primitive::Float,
    Primitive::Double,
    Primitive::Duration,
    Primitive::DateTime,
    Primitive::Time,
    Primitive::Date,
    Primitive::GYearMonth,
    Primitive::GYear,
    Primitive::GMonthDay,
    Primitive::GMonth,
    Primitive::GDay,
    Primitive::HexBinary,
    Primitive::Base64Binary,
    Primitive::AnyUri,
    Primitive::QName,
    Primitive::Notation,
];

/// One built-in derived datatype: its name, its base, and the facets that
/// derive it — each facet a `(name, value)` pair.
type Derived = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);

/// `(name, base, facets)` for every built-in *derived* atomic datatype.
const DERIVED: &[Derived] = &[
    ("normalizedString", "string", &[("whiteSpace", "replace")]),
    ("token", "normalizedString", &[("whiteSpace", "collapse")]),
    (
        "language",
        "token",
        &[("pattern", "[a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*")],
    ),
    ("Name", "token", &[("pattern", r"\i\c*")]),
    ("NMTOKEN", "token", &[("pattern", r"\c+")]),
    ("NCName", "Name", &[("pattern", r"[\i-[:]][\c-[:]]*")]),
    ("ID", "NCName", &[]),
    ("IDREF", "NCName", &[]),
    ("ENTITY", "NCName", &[]),
    (
        "integer",
        "decimal",
        &[("fractionDigits", "0"), ("pattern", "[\\-+]?[0-9]+")],
    ),
    ("nonPositiveInteger", "integer", &[("maxInclusive", "0")]),
    (
        "negativeInteger",
        "nonPositiveInteger",
        &[("maxInclusive", "-1")],
    ),
    (
        "long",
        "integer",
        &[
            ("minInclusive", "-9223372036854775808"),
            ("maxInclusive", "9223372036854775807"),
        ],
    ),
    (
        "int",
        "long",
        &[
            ("minInclusive", "-2147483648"),
            ("maxInclusive", "2147483647"),
        ],
    ),
    (
        "short",
        "int",
        &[("minInclusive", "-32768"), ("maxInclusive", "32767")],
    ),
    (
        "byte",
        "short",
        &[("minInclusive", "-128"), ("maxInclusive", "127")],
    ),
    ("nonNegativeInteger", "integer", &[("minInclusive", "0")]),
    (
        "unsignedLong",
        "nonNegativeInteger",
        &[("maxInclusive", "18446744073709551615")],
    ),
    (
        "unsignedInt",
        "unsignedLong",
        &[("maxInclusive", "4294967295")],
    ),
    ("unsignedShort", "unsignedInt", &[("maxInclusive", "65535")]),
    ("unsignedByte", "unsignedShort", &[("maxInclusive", "255")]),
    (
        "positiveInteger",
        "nonNegativeInteger",
        &[("minInclusive", "1")],
    ),
];

/// The 1.1-only derived atomic datatypes.
const DERIVED_1_1: &[Derived] = &[
    (
        "dateTimeStamp",
        "dateTime",
        &[("explicitTimezone", "required")],
    ),
    ("yearMonthDuration", "duration", &[("pattern", "[^DT]*")]),
    (
        "dayTimeDuration",
        "duration",
        &[("pattern", "[^YM]*(T.*)?")],
    ),
];

/// `(name, itemType)` for the three built-in list datatypes.
const LISTS: &[(&str, &str)] = &[
    ("IDREFS", "IDREF"),
    ("ENTITIES", "ENTITY"),
    ("NMTOKENS", "NMTOKEN"),
];

/// Builds a [`Schema`] containing nothing but the built-in components.
///
/// # Errors
///
/// Only if a built-in facet fails to compile, which cannot happen for the
/// table above; the signature keeps the construction honest.
pub(crate) fn build(version: XsdVersion) -> Result<Schema, SchemaError> {
    // The hierarchy is identical for every schema, and building it compiles a
    // handful of Unicode-aware patterns, so it is built once per version and
    // handed out by clone.
    let cell = match version {
        XsdVersion::V1_0 => &BUILTIN_1_0,
        XsdVersion::V1_1 => &BUILTIN_1_1,
    };
    if let Some(schema) = cell.with(|cached| cached.borrow().clone()) {
        return Ok(schema);
    }
    let schema = construct(version)?;
    cell.with(|cached| *cached.borrow_mut() = Some(schema.clone()));
    Ok(schema)
}

thread_local! {
    /// The XSD 1.0 built-in hierarchy, built at most once per thread.
    static BUILTIN_1_0: RefCell<Option<Schema>> = const { RefCell::new(None) };
    /// The XSD 1.1 built-in hierarchy, built at most once per thread.
    static BUILTIN_1_1: RefCell<Option<Schema>> = const { RefCell::new(None) };
}

fn construct(version: XsdVersion) -> Result<Schema, SchemaError> {
    let mut schema = Schema {
        types: Vec::new(),
        elements: Vec::new(),
        attributes: Vec::new(),
        groups: Vec::new(),
        attribute_groups: Vec::new(),
        notations: Vec::new(),
        identity_constraints: Vec::new(),
        global_types: HashMap::new(),
        global_elements: HashMap::new(),
        global_attributes: HashMap::new(),
        global_groups: HashMap::new(),
        global_attribute_groups: HashMap::new(),
        global_notations: HashMap::new(),
        global_identity_constraints: HashMap::new(),
        any_type: TypeId(0),
        any_simple_type: TypeId(1),
        version,
    };

    // `xs:anyType`: mixed content with a lax wildcard, and a lax attribute
    // wildcard. It is its own base.
    let any_particle = Particle {
        min_occurs: 1,
        max_occurs: MaxOccurs::Bounded(1),
        term: Term::Group(ModelGroup {
            compositor: Compositor::Sequence,
            particles: vec![Particle {
                min_occurs: 0,
                max_occurs: MaxOccurs::Unbounded,
                term: Term::Wildcard(Wildcard {
                    namespace: NamespaceConstraint::Any,
                    process_contents: ProcessContents::Lax,
                    disallowed: crate::components::DisallowedNames::default(),
                }),
            }],
        }),
    };
    let content_model = ContentModel::build(&any_particle)?;
    let any_type = push_type(
        &mut schema,
        TypeDefinition {
            name: Some(ExpandedName::new(Some(XSD_NS), "anyType")),
            base: None,
            derivation: Derivation::Restriction,
            final_: DerivationSet::empty(),
            block: DerivationSet::empty(),
            kind: TypeKind::Complex(Box::new(ComplexTypeInfo {
                derivation: Derivation::Restriction,
                abstract_: false,
                attribute_uses: Vec::new(),
                attribute_wildcard: Some(Wildcard {
                    namespace: NamespaceConstraint::Any,
                    process_contents: ProcessContents::Lax,
                    disallowed: crate::components::DisallowedNames::default(),
                }),
                content: ContentType::Mixed(Box::new(any_particle)),
                content_model: Some(content_model),
                assertions: Vec::new(),
                open_content: None,
            })),
        },
    );
    schema.any_type = any_type;

    // `xs:anySimpleType`.
    let any_simple_type = push_type(
        &mut schema,
        TypeDefinition {
            name: Some(ExpandedName::new(Some(XSD_NS), "anySimpleType")),
            base: Some(any_type),
            derivation: Derivation::Restriction,
            final_: DerivationSet::empty(),
            block: DerivationSet::empty(),
            kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                variety: Variety::Absent,
                primitive: None,
                facets: Facets {
                    white_space: WhiteSpace::Preserve,
                    white_space_fixed: false,
                    ..Facets::default()
                },
            })),
        },
    );
    schema.any_simple_type = any_simple_type;

    let atomic_root = if version.is_1_1() {
        push_type(
            &mut schema,
            TypeDefinition {
                name: Some(ExpandedName::new(Some(XSD_NS), "anyAtomicType")),
                base: Some(any_simple_type),
                derivation: Derivation::Restriction,
                // XSD 1.1 §3.16.7.2: `xs:anyAtomicType` is a *special* type
                // that only the built-in primitives derive from — a
                // user-defined *simple* type may not restrict it, list it or
                // unite it (`saxonData/Simple/simple051` … `simple053`). A
                // complex type with simple content may still extend it
                // (`ibmData/valid/D4_3_15/d4_3_15v19`), so `extension` stays
                // out of the set; the built-in primitives below are pushed
                // straight into the arena and so are unaffected either way.
                final_: DerivationSet {
                    extension: false,
                    restriction: true,
                    list: true,
                    union: true,
                    substitution: false,
                },
                block: DerivationSet::empty(),
                kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                    variety: Variety::Absent,
                    primitive: None,
                    facets: Facets {
                        white_space: WhiteSpace::Preserve,
                        white_space_fixed: false,
                        ..Facets::default()
                    },
                })),
            },
        )
    } else {
        any_simple_type
    };

    for primitive in PRIMITIVES {
        let white_space = primitive.white_space();
        push_type(
            &mut schema,
            TypeDefinition {
                name: Some(ExpandedName::new(Some(XSD_NS), primitive.name())),
                base: Some(atomic_root),
                derivation: Derivation::Restriction,
                final_: DerivationSet::empty(),
                block: DerivationSet::empty(),
                kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                    variety: Variety::Atomic(*primitive),
                    primitive: Some(*primitive),
                    facets: Facets {
                        white_space,
                        // Only `xs:string` leaves `whiteSpace` free to be
                        // narrowed; every other primitive fixes `collapse`.
                        white_space_fixed: !matches!(primitive, Primitive::String),
                        ..Facets::default()
                    },
                })),
            },
        );
    }

    let derived: Vec<&Derived> = if version.is_1_1() {
        DERIVED.iter().chain(DERIVED_1_1.iter()).collect()
    } else {
        DERIVED.iter().collect()
    };
    for (name, base_name, facet_specs) in derived {
        let base = lookup(&schema, base_name)?;
        let base_definition = schema.type_definition(base);
        let base_simple = base_definition.as_simple().ok_or_else(|| {
            SchemaError::new("st-props-correct", format!("`{base_name}` is not simple"))
        })?;
        let variety = base_simple.variety.clone();
        let primitive = base_simple.primitive;
        let base_facets = base_simple.facets.clone();
        let raw: Vec<RawFacet> = facet_specs
            .iter()
            .map(|(facet, value)| RawFacet {
                name: (*facet).to_owned(),
                value: (*value).to_owned(),
                // The Part 2 schema for the datatypes marks none of these
                // facets `fixed`; the ordering rule alone stops a user
                // restriction from weakening them.
                fixed: false,
                namespaces: NamespaceContext::new(),
                xpath_default_namespace: None,
            })
            .collect();
        let facets = apply_facets(&schema, base, &base_facets, &variety, &raw)?;
        push_type(
            &mut schema,
            TypeDefinition {
                name: Some(ExpandedName::new(Some(XSD_NS), *name)),
                base: Some(base),
                derivation: Derivation::Restriction,
                final_: DerivationSet::empty(),
                block: DerivationSet::empty(),
                kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                    variety,
                    primitive,
                    facets,
                })),
            },
        );
    }

    for (name, item_name) in LISTS {
        let item = lookup(&schema, item_name)?;
        push_type(
            &mut schema,
            TypeDefinition {
                name: Some(ExpandedName::new(Some(XSD_NS), *name)),
                base: Some(any_simple_type),
                derivation: Derivation::List,
                final_: DerivationSet::empty(),
                block: DerivationSet::empty(),
                kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                    variety: Variety::List(item),
                    primitive: None,
                    facets: Facets {
                        white_space: WhiteSpace::Collapse,
                        white_space_fixed: true,
                        min_length: Some(1),
                        ..Facets::default()
                    },
                })),
            },
        );
    }

    if version.is_1_1() {
        // XSD 1.1 Part 2 §3.2: `xs:error` is a union with no member types, so
        // no value at all is valid against it. It exists to make a branch of a
        // conditional type assignment (or of a substitution group) an error.
        push_type(
            &mut schema,
            TypeDefinition {
                name: Some(ExpandedName::new(Some(XSD_NS), "error")),
                base: Some(any_simple_type),
                derivation: Derivation::Union,
                final_: DerivationSet::empty(),
                block: DerivationSet::empty(),
                kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                    variety: Variety::Union(Vec::new()),
                    primitive: None,
                    facets: Facets::default(),
                })),
            },
        );
    }

    declare_xml_attributes(&mut schema)?;
    declare_instance_attributes(&mut schema)?;

    Ok(schema)
}

/// The four attributes of the *XML Schema instance* namespace — `xsi:type`,
/// `xsi:nil`, `xsi:schemaLocation` and `xsi:noNamespaceSchemaLocation`.
///
/// §3.2.7 declares them in the schema for schema documents, and §4.2.6 makes
/// them present in *every* schema, so a schema document may refer to them with
/// `<xs:attribute ref="xsi:type"/>` without importing anything.
fn declare_instance_attributes(schema: &mut Schema) -> Result<(), SchemaError> {
    let qname = lookup(schema, "QName")?;
    let boolean = lookup(schema, "boolean")?;
    let any_uri = lookup(schema, "anyURI")?;
    // `xsi:schemaLocation` is a list of `xs:anyURI`, which `xs:ENTITIES` and
    // friends do not model; the list built here has no name of its own.
    let locations = push_type(
        schema,
        TypeDefinition {
            name: None,
            base: Some(schema.any_simple_type),
            derivation: Derivation::List,
            final_: DerivationSet::empty(),
            block: DerivationSet::empty(),
            kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                variety: Variety::List(any_uri),
                primitive: None,
                facets: Facets {
                    white_space: WhiteSpace::Collapse,
                    white_space_fixed: true,
                    ..Facets::default()
                },
            })),
        },
    );
    for (local, type_id) in [
        ("type", qname),
        ("nil", boolean),
        ("schemaLocation", locations),
        ("noNamespaceSchemaLocation", any_uri),
    ] {
        let name = ExpandedName::new(Some(XSI_NS), local);
        let attribute = AttributeId(u32::try_from(schema.attributes.len()).unwrap_or(u32::MAX));
        schema.attributes.push(AttributeDeclaration {
            name: name.clone(),
            type_id,
            global: true,
            value_constraint: None,
            inheritable: false,
        });
        schema.global_attributes.insert(name, attribute);
    }
    Ok(())
}

/// The four attributes of [the schema for the XML namespace][xml-xsd], which
/// every processor knows without an `xs:import`: `xml:lang`, `xml:space`,
/// `xml:base` and `xml:id`, plus the `xml:specialAttrs` attribute group.
///
/// The declarations here are deliberately permissive — `xml:lang` is typed
/// `xs:string` rather than the union of `xs:language` and the empty string —
/// because their purpose is to let schemas *refer* to them, not to police the
/// XML specification's own attributes.
///
/// [xml-xsd]: https://www.w3.org/2001/xml.xsd
fn declare_xml_attributes(schema: &mut Schema) -> Result<(), SchemaError> {
    let string = lookup(schema, "string")?;
    let ncname = lookup(schema, "NCName")?;
    let any_uri = lookup(schema, "anyURI")?;
    let id = lookup(schema, "ID")?;
    let mut uses = Vec::new();
    for (local, type_id, in_group) in [
        ("lang", string, true),
        ("space", ncname, true),
        ("base", any_uri, true),
        ("id", id, false),
    ] {
        let name = ExpandedName::new(Some(XML_NS), local);
        let attribute = AttributeId(u32::try_from(schema.attributes.len()).unwrap_or(u32::MAX));
        schema.attributes.push(AttributeDeclaration {
            name: name.clone(),
            type_id,
            global: true,
            value_constraint: None,
            inheritable: false,
        });
        schema.global_attributes.insert(name, attribute);
        if in_group {
            uses.push(AttributeUse {
                attribute,
                kind: AttributeUseKind::Optional,
                value_constraint: None,
                inheritable: false,
            });
        }
    }
    let group_name = ExpandedName::new(Some(XML_NS), "specialAttrs");
    let group = AttrGroupId(u32::try_from(schema.attribute_groups.len()).unwrap_or(u32::MAX));
    schema.attribute_groups.push(AttributeGroupDefinition {
        name: Some(group_name.clone()),
        attribute_uses: uses,
        attribute_wildcard: None,
    });
    schema.global_attribute_groups.insert(group_name, group);
    Ok(())
}

fn push_type(schema: &mut Schema, definition: TypeDefinition) -> TypeId {
    let id = TypeId(u32::try_from(schema.types.len()).unwrap_or(u32::MAX));
    if let Some(name) = definition.name.clone() {
        schema.global_types.insert(name, id);
    }
    schema.types.push(definition);
    id
}

fn lookup(schema: &Schema, local: &str) -> Result<TypeId, SchemaError> {
    schema.builtin(local).ok_or_else(|| {
        SchemaError::new(
            "src-resolve",
            format!("the built-in datatype `{local}` is missing"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hierarchy_is_complete() -> Result<(), SchemaError> {
        let schema = build(XsdVersion::V1_0)?;
        // 19 primitives + 2 ur-types + 23 derived + 3 lists, plus the
        // anonymous list type behind `xsi:schemaLocation`.
        assert_eq!(
            schema.type_count(),
            19 + 2 + DERIVED.len() + LISTS.len() + 1
        );
        assert!(schema.find_attribute(Some(XSI_NS), "type").is_some());
        assert!(schema.find_attribute(Some(XSI_NS), "nil").is_some());
        assert!(schema.builtin("string").is_some());
        assert!(schema.builtin("anyAtomicType").is_none());
        Ok(())
    }

    #[test]
    fn version_1_1_adds_its_datatypes() -> Result<(), SchemaError> {
        let schema = build(XsdVersion::V1_1)?;
        assert!(schema.builtin("anyAtomicType").is_some());
        assert!(schema.builtin("dateTimeStamp").is_some());
        assert!(schema.builtin("dayTimeDuration").is_some());
        // `xs:error` exists only in 1.1, and nothing is valid against it.
        assert!(schema.builtin("error").is_some());
        assert!(build(XsdVersion::V1_0)?.builtin("error").is_none());
        Ok(())
    }

    #[test]
    fn derivation_chains_link_up() -> Result<(), SchemaError> {
        let schema = build(XsdVersion::V1_0)?;
        let byte = schema.builtin("byte").expect("byte");
        let decimal = schema.builtin("decimal").expect("decimal");
        assert!(schema.is_derived_from(byte, decimal));
        assert!(!schema.is_derived_from(decimal, byte));
        Ok(())
    }
}
