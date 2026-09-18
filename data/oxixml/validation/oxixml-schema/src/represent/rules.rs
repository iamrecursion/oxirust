//! The schema for schemas, as a table.
//!
//! Every element of the schema vocabulary gets one [`Rule`]: the attributes it
//! may carry (each with the datatype [`values`](super::values) checks), the
//! attributes it must carry, and its **content model** — the ordered,
//! bounded sequence of children [XML Schema Part 1
//! §3.2](https://www.w3.org/TR/xmlschema-1/#normative-schemaSchema) declares
//! for it.
//!
//! Modelling the content as an ordered sequence rather than a set is what
//! catches the large family of schema documents that put an `xs:attribute`
//! before the model group, repeat `xs:anyAttribute`, or write both an
//! `xs:simpleContent` and an `xs:complexContent` inside one `xs:complexType`.

use super::values::AttrType;

/// Where an element of the vocabulary sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Context {
    /// A child of `xs:schema`, `xs:redefine` or `xs:override`.
    TopLevel,
    /// Anywhere else — except the two cases below, which change the model.
    Local,
    /// Inside `xs:simpleContent`.
    SimpleContent,
    /// Inside `xs:complexContent`.
    ComplexContent,
    /// Inside `xs:simpleType`.
    SimpleType,
}

/// One term of a content model: how many children out of a name set may occur
/// at this point.
#[derive(Debug, Clone, Copy)]
pub(super) struct Term {
    /// The local names this term admits.
    pub names: &'static [&'static str],
    /// The least number of children it must consume.
    pub min: u32,
    /// The greatest number it may consume; [`u32::MAX`] is `unbounded`.
    pub max: u32,
}

const fn opt(names: &'static [&'static str]) -> Term {
    Term {
        names,
        min: 0,
        max: 1,
    }
}

const fn one(names: &'static [&'static str]) -> Term {
    Term {
        names,
        min: 1,
        max: 1,
    }
}

const fn star(names: &'static [&'static str]) -> Term {
    Term {
        names,
        min: 0,
        max: u32::MAX,
    }
}

const fn plus(names: &'static [&'static str]) -> Term {
    Term {
        names,
        min: 1,
        max: u32::MAX,
    }
}

/// A content model: a choice between sequences of [`Term`]s.
///
/// The alternatives are tried in order and the first that consumes every child
/// wins. Within one alternative the terms are matched greedily, which is exact
/// because the name sets of consecutive terms in the schema for schemas are
/// disjoint.
pub(super) type Model = &'static [&'static [Term]];

const M_SCHEMA: Model = &[&[
    star(&["include", "import", "redefine", "override"]),
    opt(&["defaultOpenContent"]),
    star(&[
        "simpleType",
        "complexType",
        "group",
        "attributeGroup",
        "element",
        "attribute",
        "notation",
    ]),
]];

const M_ELEMENT: Model = &[&[opt(TYPES), star(&["alternative"]), star(IDENTITY)]];

const M_ATTRIBUTE: Model = &[&[opt(SIMPLE_TYPE)]];

const M_COMPLEX_TYPE: Model = &[
    &[one(&["simpleContent"])],
    &[one(&["complexContent"])],
    &[
        opt(OPEN_CONTENT),
        opt(GROUPS),
        star(ATTRIBUTES),
        opt(ANY_ATTRIBUTE),
        star(ASSERT),
    ],
];

const M_SIMPLE_TYPE: Model = &[&[one(&["restriction", "list", "union"])]];

const M_SIMPLE_RESTRICTION: Model = &[&[opt(SIMPLE_TYPE), star(FACETS)]];

const M_SIMPLE_CONTENT_RESTRICTION: Model = &[&[
    opt(SIMPLE_TYPE),
    star(FACETS),
    star(ATTRIBUTES),
    opt(ANY_ATTRIBUTE),
    star(ASSERT),
]];

const M_COMPLEX_DERIVATION: Model = &[&[
    opt(OPEN_CONTENT),
    opt(GROUPS),
    star(ATTRIBUTES),
    opt(ANY_ATTRIBUTE),
    star(ASSERT),
]];

const M_ATTRIBUTE_TAIL: Model = &[ATTRIBUTE_TAIL];

const M_DERIVATION_CHOICE: Model = &[&[one(&["restriction", "extension"])]];

const M_GROUP_DEF: Model = &[&[one(&["all", "choice", "sequence"])]];

const M_EMPTY: Model = &[&[]];

const M_ALL: Model = &[&[star(&["element", "any", "group"])]];

const M_PARTICLES: Model = &[&[star(PARTICLES)]];

const M_ATTRIBUTE_GROUP_DEF: Model = &[&[star(ATTRIBUTES), opt(ANY_ATTRIBUTE)]];

const M_IDENTITY: Model = &[&[one(&["selector"]), plus(&["field"])], &[]];

const M_REDEFINE: Model = &[&[star(&[
    "simpleType",
    "complexType",
    "group",
    "attributeGroup",
])]];

const M_OVERRIDE: Model = &[&[star(&[
    "simpleType",
    "complexType",
    "group",
    "attributeGroup",
    "element",
    "attribute",
    "notation",
])]];

const M_ANNOTATION: Model = &[&[star(&["appinfo", "documentation"])]];

const M_UNION: Model = &[&[star(SIMPLE_TYPE)]];

const M_ALTERNATIVE: Model = &[&[opt(TYPES)]];

const M_OPEN_CONTENT: Model = &[&[opt(&["any"])]];

/// What one element of the schema vocabulary allows.
pub(super) struct Rule {
    /// Unqualified attributes it may carry, beyond `id`, with their datatypes.
    pub attributes: &'static [(&'static str, AttrType)],
    /// Unqualified attributes it must carry.
    pub required: &'static [&'static str],
    /// Its content model, with any `xs:annotation` already removed.
    pub content: Model,
}

/// The constraining facets, which all share a shape.
pub(super) const FACETS: &[&str] = &[
    "length",
    "minLength",
    "maxLength",
    "pattern",
    "enumeration",
    "whiteSpace",
    "maxInclusive",
    "maxExclusive",
    "minInclusive",
    "minExclusive",
    "totalDigits",
    "fractionDigits",
    "assertion",
    "explicitTimezone",
];

/// The 1.1-only elements, rejected outright under 1.0.
pub(super) const ONLY_1_1: &[&str] = &[
    "assert",
    "alternative",
    "openContent",
    "defaultOpenContent",
    "override",
];

const GROUPS: &[&str] = &["group", "all", "choice", "sequence"];
const PARTICLES: &[&str] = &["element", "group", "choice", "sequence", "any"];
const ATTRIBUTES: &[&str] = &["attribute", "attributeGroup"];
const ANY_ATTRIBUTE: &[&str] = &["anyAttribute"];
const IDENTITY: &[&str] = &["unique", "key", "keyref"];
const TYPES: &[&str] = &["simpleType", "complexType"];
const OPEN_CONTENT: &[&str] = &["openContent"];
const ASSERT: &[&str] = &["assert"];
const SIMPLE_TYPE: &[&str] = &["simpleType"];

/// `((attribute | attributeGroup)*, anyAttribute?, assert*)`
const ATTRIBUTE_TAIL: &[Term] = &[star(ATTRIBUTES), opt(ANY_ATTRIBUTE), star(ASSERT)];

/// The `#all`-or-list vocabularies of `block`, `final` and their defaults.
const BLOCK_SET: &[&str] = &["extension", "restriction", "substitution"];
const TYPE_DERIVATION: &[&str] = &["extension", "restriction"];
const FULL_DERIVATION: &[&str] = &["extension", "restriction", "list", "union"];
const SIMPLE_DERIVATION: &[&str] = &["list", "union", "restriction"];

const FORM: &[&str] = &["qualified", "unqualified"];
const USE: &[&str] = &["prohibited", "optional", "required"];
const PROCESS: &[&str] = &["skip", "lax", "strict"];
const MODE: &[&str] = &["none", "interleave", "suffix"];
const WHITE_SPACE: &[&str] = &["preserve", "replace", "collapse"];
const TIMEZONE: &[&str] = &["required", "prohibited", "optional"];

/// The rule for one element of the vocabulary, or [`None`] when the name is
/// not part of it at all.
#[allow(clippy::too_many_lines)]
pub(super) fn rule(name: &str, context: Context) -> Option<Rule> {
    let top = context == Context::TopLevel;
    Some(match (name, context) {
        ("schema", _) => Rule {
            attributes: &[
                ("targetNamespace", AttrType::AnyUri),
                ("version", AttrType::Token),
                ("finalDefault", AttrType::DerivationSet(FULL_DERIVATION)),
                ("blockDefault", AttrType::DerivationSet(BLOCK_SET)),
                ("attributeFormDefault", AttrType::Enum(FORM)),
                ("elementFormDefault", AttrType::Enum(FORM)),
                ("defaultAttributes", AttrType::QName),
                ("xpathDefaultNamespace", AttrType::Token),
            ],
            required: &[],
            // `((include | import | redefine | override | annotation)*,
            //   (defaultOpenContent, annotation*)?,
            //   ((simpleType | complexType | group | attributeGroup |
            //     element | attribute | notation), annotation*)*)`
            content: M_SCHEMA,
        },
        ("element", Context::TopLevel) => Rule {
            attributes: &[
                ("name", AttrType::NcName),
                ("type", AttrType::QName),
                ("substitutionGroup", AttrType::QNameList),
                ("default", AttrType::Text),
                ("fixed", AttrType::Text),
                ("nillable", AttrType::Boolean),
                ("abstract", AttrType::Boolean),
                ("final", AttrType::DerivationSet(TYPE_DERIVATION)),
                ("block", AttrType::DerivationSet(BLOCK_SET)),
            ],
            required: &["name"],
            content: M_ELEMENT,
        },
        ("element", _) => Rule {
            attributes: &[
                ("name", AttrType::NcName),
                ("ref", AttrType::QName),
                ("type", AttrType::QName),
                ("default", AttrType::Text),
                ("fixed", AttrType::Text),
                ("form", AttrType::Enum(FORM)),
                ("minOccurs", AttrType::NonNegativeInteger),
                ("maxOccurs", AttrType::MaxOccurs),
                ("nillable", AttrType::Boolean),
                ("block", AttrType::DerivationSet(BLOCK_SET)),
                ("targetNamespace", AttrType::AnyUri),
            ],
            required: &[],
            content: M_ELEMENT,
        },
        ("attribute", Context::TopLevel) => Rule {
            attributes: &[
                ("name", AttrType::NcName),
                ("type", AttrType::QName),
                ("default", AttrType::Text),
                ("fixed", AttrType::Text),
                ("inheritable", AttrType::Boolean),
            ],
            required: &["name"],
            content: M_ATTRIBUTE,
        },
        ("attribute", _) => Rule {
            attributes: &[
                ("name", AttrType::NcName),
                ("ref", AttrType::QName),
                ("type", AttrType::QName),
                ("use", AttrType::Enum(USE)),
                ("default", AttrType::Text),
                ("fixed", AttrType::Text),
                ("form", AttrType::Enum(FORM)),
                ("inheritable", AttrType::Boolean),
                ("targetNamespace", AttrType::AnyUri),
            ],
            required: &[],
            content: M_ATTRIBUTE,
        },
        ("complexType", _) => Rule {
            attributes: if top {
                &[
                    ("name", AttrType::NcName),
                    ("abstract", AttrType::Boolean),
                    ("final", AttrType::DerivationSet(TYPE_DERIVATION)),
                    ("block", AttrType::DerivationSet(TYPE_DERIVATION)),
                    ("mixed", AttrType::Boolean),
                    ("defaultAttributesApply", AttrType::Boolean),
                ]
            } else {
                &[
                    ("mixed", AttrType::Boolean),
                    ("defaultAttributesApply", AttrType::Boolean),
                ]
            },
            required: if top { &["name"] } else { &[] },
            // `(simpleContent | complexContent |
            //   (openContent?, (group | all | choice | sequence)?,
            //    ((attribute | attributeGroup)*, anyAttribute?), assert*))`
            content: M_COMPLEX_TYPE,
        },
        ("simpleType", _) => Rule {
            attributes: if top {
                &[
                    ("name", AttrType::NcName),
                    ("final", AttrType::DerivationSet(SIMPLE_DERIVATION)),
                ]
            } else {
                &[("final", AttrType::DerivationSet(SIMPLE_DERIVATION))]
            },
            required: if top { &["name"] } else { &[] },
            content: M_SIMPLE_TYPE,
        },
        ("restriction", Context::SimpleType) => Rule {
            attributes: &[("base", AttrType::QName)],
            required: &[],
            content: M_SIMPLE_RESTRICTION,
        },
        ("restriction", Context::SimpleContent) => Rule {
            attributes: &[("base", AttrType::QName)],
            required: &[],
            content: M_SIMPLE_CONTENT_RESTRICTION,
        },
        ("restriction", _) => Rule {
            attributes: &[("base", AttrType::QName)],
            required: &[],
            content: M_COMPLEX_DERIVATION,
        },
        ("extension", Context::SimpleContent) => Rule {
            attributes: &[("base", AttrType::QName)],
            required: &["base"],
            content: M_ATTRIBUTE_TAIL,
        },
        ("extension", _) => Rule {
            attributes: &[("base", AttrType::QName)],
            required: &["base"],
            content: M_COMPLEX_DERIVATION,
        },
        ("simpleContent", _) => Rule {
            attributes: &[],
            required: &[],
            content: M_DERIVATION_CHOICE,
        },
        ("complexContent", _) => Rule {
            attributes: &[("mixed", AttrType::Boolean)],
            required: &[],
            content: M_DERIVATION_CHOICE,
        },
        ("group", Context::TopLevel) => Rule {
            attributes: &[("name", AttrType::NcName)],
            required: &["name"],
            content: M_GROUP_DEF,
        },
        ("group", _) => Rule {
            attributes: &[
                ("ref", AttrType::QName),
                ("minOccurs", AttrType::NonNegativeInteger),
                ("maxOccurs", AttrType::MaxOccurs),
            ],
            required: &["ref"],
            content: M_EMPTY,
        },
        ("all", _) => Rule {
            attributes: &[
                ("minOccurs", AttrType::NonNegativeInteger),
                ("maxOccurs", AttrType::MaxOccurs),
            ],
            required: &[],
            // 1.1 lets `xs:all` hold `xs:any` and `xs:group` too; the 1.0-only
            // restriction to `xs:element` is enforced by `check_all_group`.
            content: M_ALL,
        },
        ("choice" | "sequence", _) => Rule {
            attributes: &[
                ("minOccurs", AttrType::NonNegativeInteger),
                ("maxOccurs", AttrType::MaxOccurs),
            ],
            required: &[],
            content: M_PARTICLES,
        },
        ("any", _) => Rule {
            attributes: &[
                ("minOccurs", AttrType::NonNegativeInteger),
                ("maxOccurs", AttrType::MaxOccurs),
                ("namespace", AttrType::NamespaceList),
                ("processContents", AttrType::Enum(PROCESS)),
                ("notNamespace", AttrType::NotNamespaceList),
                ("notQName", AttrType::NotQNameList),
            ],
            required: &[],
            content: M_EMPTY,
        },
        ("anyAttribute", _) => Rule {
            attributes: &[
                ("namespace", AttrType::NamespaceList),
                ("processContents", AttrType::Enum(PROCESS)),
                ("notNamespace", AttrType::NotNamespaceList),
                ("notQName", AttrType::NotQNameList),
            ],
            required: &[],
            content: M_EMPTY,
        },
        ("attributeGroup", Context::TopLevel) => Rule {
            attributes: &[("name", AttrType::NcName)],
            required: &["name"],
            content: M_ATTRIBUTE_GROUP_DEF,
        },
        ("attributeGroup", _) => Rule {
            attributes: &[("ref", AttrType::QName)],
            required: &["ref"],
            content: M_EMPTY,
        },
        ("unique" | "key", _) => Rule {
            attributes: &[("name", AttrType::NcName), ("ref", AttrType::QName)],
            required: &[],
            content: M_IDENTITY,
        },
        ("keyref", _) => Rule {
            attributes: &[
                ("name", AttrType::NcName),
                ("refer", AttrType::QName),
                ("ref", AttrType::QName),
            ],
            required: &[],
            content: M_IDENTITY,
        },
        ("selector" | "field", _) => Rule {
            attributes: &[
                ("xpath", AttrType::XPath),
                ("xpathDefaultNamespace", AttrType::Token),
            ],
            required: &["xpath"],
            content: M_EMPTY,
        },
        ("notation", _) => Rule {
            attributes: &[
                ("name", AttrType::NcName),
                ("public", AttrType::Token),
                ("system", AttrType::AnyUri),
            ],
            required: &["name"],
            content: M_EMPTY,
        },
        ("include", _) => Rule {
            attributes: &[("schemaLocation", AttrType::AnyUri)],
            required: &["schemaLocation"],
            content: M_EMPTY,
        },
        ("import", _) => Rule {
            attributes: &[
                ("namespace", AttrType::AnyUri),
                ("schemaLocation", AttrType::AnyUri),
            ],
            required: &[],
            content: M_EMPTY,
        },
        ("redefine", _) => Rule {
            attributes: &[("schemaLocation", AttrType::AnyUri)],
            required: &["schemaLocation"],
            content: M_REDEFINE,
        },
        ("override", _) => Rule {
            attributes: &[("schemaLocation", AttrType::AnyUri)],
            required: &["schemaLocation"],
            content: M_OVERRIDE,
        },
        ("annotation", _) => Rule {
            attributes: &[],
            required: &[],
            content: M_ANNOTATION,
        },
        ("appinfo" | "documentation", _) => Rule {
            attributes: &[("source", AttrType::AnyUri)],
            required: &[],
            content: M_EMPTY,
        },
        ("list", _) => Rule {
            attributes: &[("itemType", AttrType::QName)],
            required: &[],
            content: M_ATTRIBUTE,
        },
        ("union", _) => Rule {
            attributes: &[("memberTypes", AttrType::QNameList)],
            required: &[],
            content: M_UNION,
        },
        ("assert" | "assertion", _) => Rule {
            attributes: &[
                ("test", AttrType::XPath),
                ("xpathDefaultNamespace", AttrType::Token),
            ],
            required: &["test"],
            content: M_EMPTY,
        },
        ("alternative", _) => Rule {
            attributes: &[
                ("test", AttrType::XPath),
                ("type", AttrType::QName),
                ("xpathDefaultNamespace", AttrType::Token),
            ],
            required: &[],
            content: M_ALTERNATIVE,
        },
        ("openContent" | "defaultOpenContent", _) => Rule {
            attributes: &[
                ("mode", AttrType::Enum(MODE)),
                ("appliesToEmpty", AttrType::Boolean),
            ],
            required: &[],
            content: M_OPEN_CONTENT,
        },
        ("enumeration" | "pattern", _) => Rule {
            attributes: &[("value", AttrType::Text)],
            required: &["value"],
            content: M_EMPTY,
        },
        ("length" | "minLength" | "maxLength" | "fractionDigits", _) => Rule {
            attributes: &[
                ("value", AttrType::NonNegativeInteger),
                ("fixed", AttrType::Boolean),
            ],
            required: &["value"],
            content: M_EMPTY,
        },
        ("totalDigits", _) => Rule {
            attributes: &[
                ("value", AttrType::PositiveInteger),
                ("fixed", AttrType::Boolean),
            ],
            required: &["value"],
            content: M_EMPTY,
        },
        ("whiteSpace", _) => Rule {
            attributes: &[
                ("value", AttrType::Enum(WHITE_SPACE)),
                ("fixed", AttrType::Boolean),
            ],
            required: &["value"],
            content: M_EMPTY,
        },
        ("explicitTimezone", _) => Rule {
            attributes: &[
                ("value", AttrType::Enum(TIMEZONE)),
                ("fixed", AttrType::Boolean),
            ],
            required: &["value"],
            content: M_EMPTY,
        },
        (name, _) if FACETS.contains(&name) => Rule {
            attributes: &[("value", AttrType::Text), ("fixed", AttrType::Boolean)],
            required: &["value"],
            content: M_EMPTY,
        },
        _ => return None,
    })
}

/// Matches `children` — already stripped of annotations — against `model`.
///
/// Returns the local name of the first child that does not fit, or [`None`]
/// when the whole sequence is accepted.
pub(super) fn matches<'a>(model: Model, children: &[&'a str]) -> Option<&'a str> {
    let mut furthest = 0;
    for alternative in model {
        match match_sequence(alternative, children) {
            Ok(()) => return None,
            Err(index) => furthest = furthest.max(index),
        }
    }
    Some(children.get(furthest).copied().unwrap_or("(end)"))
}

/// Greedily matches one alternative, returning the index of the first child it
/// could not place.
fn match_sequence(terms: &[Term], children: &[&str]) -> Result<(), usize> {
    let mut index = 0;
    for term in terms {
        let mut taken = 0;
        while taken < term.max && index < children.len() && term.names.contains(&children[index]) {
            taken += 1;
            index += 1;
        }
        if taken < term.min {
            return Err(index);
        }
    }
    if index == children.len() {
        Ok(())
    } else {
        Err(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_of(name: &str, context: Context) -> Model {
        rule(name, context).map_or(&[], |rule| rule.content)
    }

    #[test]
    fn complex_types_place_attributes_after_the_model_group() {
        let model = model_of("complexType", Context::TopLevel);
        assert!(matches(model, &["sequence", "attribute", "anyAttribute"]).is_none());
        assert!(matches(model, &["attribute", "sequence"]).is_some());
        assert!(matches(model, &["anyAttribute", "anyAttribute"]).is_some());
        assert!(matches(model, &["simpleContent", "sequence"]).is_some());
        assert!(matches(model, &["simpleContent"]).is_none());
    }

    #[test]
    fn simple_type_restrictions_place_facets_after_the_inline_type() {
        let model = model_of("restriction", Context::SimpleType);
        assert!(matches(model, &["simpleType", "minLength"]).is_none());
        assert!(matches(model, &["minLength", "simpleType"]).is_some());
    }

    #[test]
    fn identity_constraints_need_a_selector_and_one_field() {
        let model = model_of("key", Context::Local);
        assert!(matches(model, &["selector", "field"]).is_none());
        assert!(matches(model, &["field", "selector"]).is_some());
        assert!(matches(model, &["selector"]).is_some());
        assert!(matches(model, &[]).is_none());
    }

    #[test]
    fn declarations_may_not_precede_includes() {
        let model = model_of("schema", Context::TopLevel);
        assert!(matches(model, &["include", "element"]).is_none());
        assert!(matches(model, &["element", "include"]).is_some());
    }
}
