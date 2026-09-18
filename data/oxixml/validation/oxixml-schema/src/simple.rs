//! Simple-type validation: the lexical mapping, the constraining facets, and
//! the merge rule that a `restriction` step applies to the facets it inherits.

use std::cmp::Ordering;

use oxixml_regex::{Regex, XsdVersion as RegexXsdVersion};

use crate::components::{Facets, TypeId, Variety};
use crate::error::SchemaError;
use crate::names::XsdVersion;
use crate::schema::Schema;
use crate::value::{
    ExplicitTimezone, NamespaceContext, Primitive, Value, WhiteSpace, parse_primitive, split_tokens,
};

/// A facet exactly as it was written in a schema document, before it is checked
/// against the base type.
#[derive(Debug, Clone)]
pub struct RawFacet {
    /// The local name of the facet element, for example `maxInclusive`.
    pub name: String,
    /// The `value` attribute.
    pub value: String,
    /// Whether it carries `fixed="true"`.
    pub fixed: bool,
    /// The namespace bindings in force where it was written.
    pub namespaces: NamespaceContext,
    /// The `xpathDefaultNamespace` in force where it was written, resolved.
    pub xpath_default_namespace: Option<String>,
}

/// Why a lexical form is not a valid value of a simple type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleTypeViolation {
    /// The `cvc-…` code of the rule that failed.
    pub code: &'static str,
    /// A human-readable explanation.
    pub message: String,
}

impl SimpleTypeViolation {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SimpleTypeViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// The outcome of validating a lexical form against a simple type.
#[derive(Debug, Clone)]
pub struct SimpleValue {
    /// The `whiteSpace`-normalized lexical form — for a union, normalized by
    /// the [`member_type`](Self::member_type) that accepted it, which is the
    /// only `whiteSpace` a union has (Datatypes §4.3.6).
    pub normalized: String,
    /// The value it denotes.
    pub value: Value,
    /// For a union, the member type that accepted it.
    pub member_type: Option<TypeId>,
}

/// Validates `lexical` against the simple type `type_id`.
///
/// # Errors
///
/// Returns the first rule that the value breaks.
pub fn validate(
    schema: &Schema,
    type_id: TypeId,
    lexical: &str,
    namespaces: &NamespaceContext,
) -> Result<SimpleValue, SimpleTypeViolation> {
    validate_impl(schema, type_id, lexical, namespaces, None)
}

/// Validates a *new restriction facet's own bound value* — `facet_name` is one
/// of `minInclusive`/`maxInclusive`/`minExclusive`/`maxExclusive` — against the
/// base type `type_id`.
///
/// This is [`validate`] with one difference: when `type_id`'s own facets
/// constrain the *same* bound in the *same* exclusive sense (a new
/// `maxExclusive` checked against the base's own `maxExclusive`, or a new
/// `minExclusive` against its own `minExclusive`), equality is accepted.
/// Restricting a bound to *exactly* the base's own bound is not a widening —
/// `{v : v < x}` restricted to itself is still `{v : v < x}` — but the
/// ordinary instance check that `validate` otherwise performs enforces the
/// *strict* inequality an instance **value** must satisfy (an instance must be
/// strictly less than `maxExclusive`), which wrongly rejects a new facet
/// **declaration** that merely restates `x` (`dateTimeStamp/d3_4_28v09`).
///
/// `minInclusive` and `maxInclusive` need no such relaxation: restating the
/// base's own inclusive bound already satisfies the ordinary (`<=`/`>=`)
/// instance check, so `validate` already accepts it.
///
/// # Errors
///
/// Returns the first rule that the value breaks.
pub(crate) fn validate_bound_facet(
    schema: &Schema,
    type_id: TypeId,
    facet_name: &str,
    lexical: &str,
    namespaces: &NamespaceContext,
) -> Result<SimpleValue, SimpleTypeViolation> {
    validate_impl(schema, type_id, lexical, namespaces, Some(facet_name))
}

fn validate_impl(
    schema: &Schema,
    type_id: TypeId,
    lexical: &str,
    namespaces: &NamespaceContext,
    relax_same_kind_bound: Option<&str>,
) -> Result<SimpleValue, SimpleTypeViolation> {
    let Some(simple) = schema.simple_type(type_id) else {
        return Err(SimpleTypeViolation::new(
            "cvc-datatype-valid.1",
            format!("`{}` is not a simple type", schema.type_name(type_id)),
        ));
    };
    let facets = &simple.facets;
    let mut normalized = facets.white_space.apply(lexical).into_owned();

    check_patterns(facets, &normalized)?;

    let (value, member_type) = match &simple.variety {
        Variety::Absent => (Value::String(normalized.clone()), None),
        Variety::Atomic(primitive) => (
            parse_primitive(*primitive, &normalized, namespaces, schema.version()).ok_or_else(
                || {
                    SimpleTypeViolation::new(
                        "cvc-datatype-valid.1.2.1",
                        format!(
                            "`{normalized}` is not a valid `{}`",
                            schema.type_name(type_id)
                        ),
                    )
                },
            )?,
            None,
        ),
        Variety::List(item) => {
            let mut items = Vec::new();
            for token in split_tokens(&normalized) {
                let element = validate(schema, *item, token, namespaces).map_err(|error| {
                    SimpleTypeViolation::new(
                        error.code,
                        format!("in list item `{token}`: {}", error.message),
                    )
                })?;
                items.push(element.value);
            }
            (Value::List(items), None)
        }
        Variety::Union(members) => {
            // Datatypes §4.3.6: `whiteSpace` "does not apply directly" to a
            // union — "the normalization behavior of union types is controlled
            // by the value of whiteSpace on that one of the memberTypes against
            // which the union is successfully validated". Each member is
            // therefore offered the string as it was written, not the union's
            // own collapse of it, and the union adopts the accepted member's
            // normalization as its own. `xs:normalizedString` then keeps a
            // trailing space that collapsing would have dropped, which is what
            // makes the `fixed="abcd edfgh "` of `msData/simpleType/stE063`
            // differ from the instance's `abcd edfgh`.
            //
            // The `pattern` facets above are still read against the union's own
            // normalization: a union's {facets} may carry `pattern`, and that
            // check is made before any member is tried, so it has no member's
            // whiteSpace to borrow.
            let mut accepted = None;
            for member in members {
                if let Ok(result) = validate(schema, *member, lexical, namespaces) {
                    accepted = Some((result.normalized, result.value, Some(*member)));
                    break;
                }
            }
            let (member_normalized, value, member_type) = accepted.ok_or_else(|| {
                SimpleTypeViolation::new(
                    "cvc-datatype-valid.1.2.3",
                    format!(
                        "`{lexical}` matches no member type of `{}`",
                        schema.type_name(type_id)
                    ),
                )
            })?;
            normalized = member_normalized;
            (value, member_type)
        }
    };

    check_facets(schema, facets, &value, relax_same_kind_bound)?;
    check_assertions(schema, facets, member_type.unwrap_or(type_id), &normalized)?;

    Ok(SimpleValue {
        normalized,
        value,
        member_type,
    })
}

/// The XSD 1.1 `assertion` facet (Part 2 §4.3.12).
///
/// Each assertion is evaluated over a fabricated element whose typed value is
/// the value being checked, with `$value` bound to that same value; an
/// assertion that does not hold — or whose evaluation raises a dynamic error —
/// makes the value invalid under *cvc-assertion*.
fn check_assertions(
    schema: &Schema,
    facets: &Facets,
    type_id: TypeId,
    normalized: &str,
) -> Result<(), SimpleTypeViolation> {
    if facets.assertions.is_empty() {
        return Ok(());
    }
    // Datatypes §4.3.13.3 clause 1.5: a value with no ·XDM representation·
    // under `T` cannot be checked and is not facet-valid.
    let Some((_, value)) = crate::xdm::facet_instance(schema, type_id, normalized) else {
        return Err(SimpleTypeViolation::new(
            "cvc-assertion",
            "the value cannot be viewed as an XDM instance, so its assertions cannot be evaluated",
        ));
    };
    for assertion in &facets.assertions {
        if !crate::assertion::evaluate_facet(assertion, value.clone()) {
            return Err(SimpleTypeViolation::new(
                "cvc-assertion",
                format!(
                    "`{normalized}` does not satisfy the assertion `{}`",
                    assertion.test
                ),
            ));
        }
    }
    Ok(())
}

fn check_patterns(facets: &Facets, normalized: &str) -> Result<(), SimpleTypeViolation> {
    for (index, group) in facets.patterns.iter().enumerate() {
        if !group.iter().any(|pattern| pattern.is_match(normalized)) {
            let sources = facets
                .pattern_sources
                .get(index)
                .map(|group| group.join("|"))
                .unwrap_or_default();
            return Err(SimpleTypeViolation::new(
                "cvc-pattern-valid",
                format!("`{normalized}` does not match the pattern `{sources}`"),
            ));
        }
    }
    Ok(())
}

fn check_facets(
    schema: &Schema,
    facets: &Facets,
    value: &Value,
    relax_same_kind_bound: Option<&str>,
) -> Result<(), SimpleTypeViolation> {
    if facets.length.is_some() || facets.min_length.is_some() || facets.max_length.is_some() {
        if let Some(measure) = value.measure() {
            if let Some(length) = facets.length
                && measure != length
            {
                return Err(SimpleTypeViolation::new(
                    "cvc-length-valid",
                    format!("the length is {measure}, but `length` is {length}"),
                ));
            }
            if let Some(min) = facets.min_length
                && measure < min
            {
                return Err(SimpleTypeViolation::new(
                    "cvc-minLength-valid",
                    format!("the length is {measure}, but `minLength` is {min}"),
                ));
            }
            if let Some(max) = facets.max_length
                && measure > max
            {
                return Err(SimpleTypeViolation::new(
                    "cvc-maxLength-valid",
                    format!("the length is {measure}, but `maxLength` is {max}"),
                ));
            }
        }
    }

    check_bound(
        value,
        facets.min_inclusive.as_ref(),
        &[Ordering::Greater, Ordering::Equal],
        "cvc-minInclusive-valid",
        "minInclusive",
    )?;
    check_bound(
        value,
        facets.max_inclusive.as_ref(),
        &[Ordering::Less, Ordering::Equal],
        "cvc-maxInclusive-valid",
        "maxInclusive",
    )?;
    check_bound(
        value,
        facets.min_exclusive.as_ref(),
        if relax_same_kind_bound == Some("minExclusive") {
            &[Ordering::Greater, Ordering::Equal]
        } else {
            &[Ordering::Greater]
        },
        "cvc-minExclusive-valid",
        "minExclusive",
    )?;
    check_bound(
        value,
        facets.max_exclusive.as_ref(),
        if relax_same_kind_bound == Some("maxExclusive") {
            &[Ordering::Less, Ordering::Equal]
        } else {
            &[Ordering::Less]
        },
        "cvc-maxExclusive-valid",
        "maxExclusive",
    )?;

    if let Some(decimal) = value.as_decimal() {
        if let Some(limit) = facets.total_digits
            && decimal.total_digits() > limit
        {
            return Err(SimpleTypeViolation::new(
                "cvc-totalDigits-valid",
                format!(
                    "`{decimal}` has {} significant digits, but `totalDigits` is {limit}",
                    decimal.total_digits()
                ),
            ));
        }
        if let Some(limit) = facets.fraction_digits
            && decimal.fraction_digits() > limit
        {
            return Err(SimpleTypeViolation::new(
                "cvc-fractionDigits-valid",
                format!(
                    "`{decimal}` has {} fraction digits, but `fractionDigits` is {limit}",
                    decimal.fraction_digits()
                ),
            ));
        }
    }

    if let Some(allowed) = &facets.enumeration
        && !allowed.iter().any(|candidate| candidate == value)
    {
        return Err(SimpleTypeViolation::new(
            "cvc-enumeration-valid",
            format!(
                "the value is not one of the {} enumerated values",
                allowed.len()
            ),
        ));
    }

    if let Some(rule) = facets.explicit_timezone
        && let Some(present) = value.has_timezone()
    {
        let ok = match rule {
            ExplicitTimezone::Required => present,
            ExplicitTimezone::Prohibited => !present,
            ExplicitTimezone::Optional => true,
        };
        if !ok {
            return Err(SimpleTypeViolation::new(
                "cvc-explicitTimezone-valid",
                "the timezone offset does not satisfy `explicitTimezone`",
            ));
        }
    }

    let _ = schema;
    Ok(())
}

fn check_bound(
    value: &Value,
    bound: Option<&Value>,
    accepted: &[Ordering],
    code: &'static str,
    facet: &str,
) -> Result<(), SimpleTypeViolation> {
    let Some(bound) = bound else {
        return Ok(());
    };
    match value.partial_cmp_value(bound) {
        Some(ordering) if accepted.contains(&ordering) => Ok(()),
        _ => Err(SimpleTypeViolation::new(
            code,
            format!("the value violates `{facet}`"),
        )),
    }
}

/// Whether the facet `name` ended up with a different value in `derived` than
/// it has in `base` — the comparison behind the `fixed="true"` rule.
///
/// The comparison is by *value*, not by lexical form, so a restriction that
/// restates a fixed `minInclusive="1.0"` as `minInclusive="1"` is accepted.
fn facet_value_differs(name: &str, base: &Facets, derived: &Facets) -> bool {
    match name {
        "length" => base.length != derived.length,
        "minLength" => base.min_length != derived.min_length,
        "maxLength" => base.max_length != derived.max_length,
        "whiteSpace" => base.white_space != derived.white_space,
        "minInclusive" => base.min_inclusive != derived.min_inclusive,
        "maxInclusive" => base.max_inclusive != derived.max_inclusive,
        "minExclusive" => base.min_exclusive != derived.min_exclusive,
        "maxExclusive" => base.max_exclusive != derived.max_exclusive,
        "totalDigits" => base.total_digits != derived.total_digits,
        "fractionDigits" => base.fraction_digits != derived.fraction_digits,
        "explicitTimezone" => base.explicit_timezone != derived.explicit_timezone,
        _ => false,
    }
}

/// Selects the regular-expression grammar that goes with a schema version.
///
/// The `pattern` facet's language is the one given in the Appendix of the
/// *Datatypes* part the schema is being assembled under, and the two Appendices
/// differ: XSD 1.1 accepts a bare `-` as an ordinary character-group member
/// anywhere (`[a-a-x-x]`, XML Schema Test Suite `reH19`-`reH21`) and treats an
/// unrecognised `\p{Is…}` block name as matching every character (`\p{IsaA0-a9}`,
/// `reK88`), where XSD 1.0 rejects both. Written as an explicit match so that
/// adding a version to either enum is a compile error here rather than a silent
/// mis-selection.
const fn regex_version(version: XsdVersion) -> RegexXsdVersion {
    match version {
        XsdVersion::V1_0 => RegexXsdVersion::V1_0,
        XsdVersion::V1_1 => RegexXsdVersion::V1_1,
    }
}

/// Applies the facets of one `restriction` step on top of the facets it
/// inherits from the base type.
///
/// # Errors
///
/// Returns a `…-valid-restriction` schema error when a facet is not applicable
/// to the base type, when its value is not in the base type's value space, or
/// when it weakens a facet the base type fixed.
pub fn apply_facets(
    schema: &Schema,
    base: TypeId,
    base_facets: &Facets,
    variety: &Variety,
    raw: &[RawFacet],
) -> Result<Facets, SchemaError> {
    let mut facets = Facets {
        patterns: base_facets.patterns.clone(),
        pattern_sources: base_facets.pattern_sources.clone(),
        ..base_facets.clone()
    };
    // A restriction step contributes at most one `pattern` group (the
    // disjunction of all the `pattern` facets it writes) and at most one
    // `enumeration` set.
    let mut step_patterns: Vec<Regex> = Vec::new();
    let mut step_pattern_sources: Vec<String> = Vec::new();
    let mut step_enumeration: Vec<Value> = Vec::new();
    let mut step_enumeration_lexical: Vec<String> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();

    let primitive = match variety {
        Variety::Atomic(primitive) => Some(*primitive),
        _ => None,
    };
    let is_list = matches!(variety, Variety::List(_));

    check_facet_cooccurrence(raw, base_facets)?;

    // The facets this step supplies that the base type fixed; their values are
    // compared against the base's once the whole step has been applied, since
    // that is where each facet's value has been parsed into its value space.
    let mut overridden_fixed: Vec<&str> = Vec::new();

    for facet in raw {
        let name = facet.name.as_str();
        if !matches!(name, "pattern" | "enumeration" | "assertion") {
            if seen.contains(&name) {
                return Err(SchemaError::new(
                    "src-single-facet-value",
                    format!("the facet `{name}` may not be repeated in one restriction"),
                ));
            }
            seen.push(name);
            if base_facets.fixed.contains(name) {
                overridden_fixed.push(name);
            }
            if facet.fixed {
                facets.fixed.insert(name.to_owned());
            }
        }
        match name {
            "pattern" => {
                let regex = Regex::xsd_versioned(&facet.value, regex_version(schema.version()))
                    .map_err(|error| {
                        SchemaError::new(
                            "src-pattern",
                            format!("invalid pattern `{}`: {error}", facet.value),
                        )
                    })?;
                step_patterns.push(regex);
                step_pattern_sources.push(facet.value.clone());
            }
            "enumeration" => {
                let member =
                    validate(schema, base, &facet.value, &facet.namespaces).map_err(|error| {
                        SchemaError::new(
                            "enumeration-valid-restriction",
                            format!(
                                "the enumeration value `{}` is not in the value space of the base \
                                 type: {error}",
                                facet.value
                            ),
                        )
                    })?;
                step_enumeration.push(member.value);
                step_enumeration_lexical.push(facet.value.clone());
            }
            "whiteSpace" => {
                let requested = WhiteSpace::parse(&facet.value).ok_or_else(|| {
                    SchemaError::new(
                        "src-whiteSpace",
                        format!("`{}` is not a `whiteSpace` value", facet.value),
                    )
                })?;
                if base_facets.white_space_fixed && requested != base_facets.white_space {
                    return Err(SchemaError::new(
                        "whiteSpace-valid-restriction",
                        format!(
                            "`whiteSpace` is fixed to `{}` by the base type",
                            base_facets.white_space
                        ),
                    ));
                }
                if requested < base_facets.white_space {
                    return Err(SchemaError::new(
                        "whiteSpace-valid-restriction",
                        format!(
                            "`whiteSpace=\"{requested}\"` weakens the base type's `{}`",
                            base_facets.white_space
                        ),
                    ));
                }
                facets.white_space = requested;
                facets.white_space_fixed = facet.fixed;
                if facet.fixed {
                    facets.fixed.insert(name.to_owned());
                }
            }
            "length" | "minLength" | "maxLength" => {
                // Part 2 §4.1.5: the only facets applicable to a *union* are
                // `pattern` and `enumeration`, so a length facet on one is an
                // error (`msData/simpleType/stF018`).
                let applicable = is_list
                    || primitive.is_some_and(Primitive::is_measurable)
                    || matches!(primitive, Some(Primitive::QName | Primitive::Notation));
                if !applicable {
                    return Err(SchemaError::new(
                        "cos-applicable-facets",
                        format!("`{name}` does not apply to this datatype"),
                    ));
                }
                let count = parse_count(name, &facet.value)?;
                match name {
                    "length" => {
                        if let Some(min) = base_facets.min_length
                            && count < min
                        {
                            return Err(SchemaError::new(
                                "length-valid-restriction",
                                "`length` is below the base type's `minLength`",
                            ));
                        }
                        if let Some(max) = base_facets.max_length
                            && count > max
                        {
                            return Err(SchemaError::new(
                                "length-valid-restriction",
                                "`length` is above the base type's `maxLength`",
                            ));
                        }
                        facets.length = Some(count);
                    }
                    "minLength" => {
                        if let Some(base_min) = base_facets.min_length
                            && count < base_min
                        {
                            return Err(SchemaError::new(
                                "minLength-valid-restriction",
                                "`minLength` weakens the base type's `minLength`",
                            ));
                        }
                        facets.min_length = Some(count);
                    }
                    _ => {
                        if let Some(base_max) = base_facets.max_length
                            && count > base_max
                        {
                            return Err(SchemaError::new(
                                "maxLength-valid-restriction",
                                "`maxLength` weakens the base type's `maxLength`",
                            ));
                        }
                        facets.max_length = Some(count);
                    }
                }
            }
            "minInclusive" | "maxInclusive" | "minExclusive" | "maxExclusive" => {
                if !primitive.is_some_and(Primitive::is_ordered) {
                    return Err(SchemaError::new(
                        "cos-applicable-facets",
                        format!("`{name}` does not apply to this datatype"),
                    ));
                }
                let bound =
                    validate_bound_facet(schema, base, name, &facet.value, &facet.namespaces)
                        .map_err(|error| {
                            SchemaError::new(
                                format!("{name}-valid-restriction"),
                                format!(
                                    "the `{name}` value `{}` is not in the value space of the base \
                                 type: {error}",
                                    facet.value
                                ),
                            )
                        })?
                        .value;
                match name {
                    "minInclusive" => {
                        facets.min_inclusive = Some(bound);
                        facets.min_exclusive = None;
                    }
                    "maxInclusive" => {
                        facets.max_inclusive = Some(bound);
                        facets.max_exclusive = None;
                    }
                    "minExclusive" => {
                        facets.min_exclusive = Some(bound);
                        facets.min_inclusive = None;
                    }
                    _ => {
                        facets.max_exclusive = Some(bound);
                        facets.max_inclusive = None;
                    }
                }
            }
            "totalDigits" | "fractionDigits" => {
                if !primitive.is_some_and(Primitive::is_numeric_decimal) {
                    return Err(SchemaError::new(
                        "cos-applicable-facets",
                        format!("`{name}` does not apply to this datatype"),
                    ));
                }
                let count = parse_count(name, &facet.value)?;
                if name == "totalDigits" {
                    if count == 0 {
                        return Err(SchemaError::new(
                            "totalDigits-valid-restriction",
                            "`totalDigits` must be a positive integer",
                        ));
                    }
                    if let Some(base_limit) = base_facets.total_digits
                        && count > base_limit
                    {
                        return Err(SchemaError::new(
                            "totalDigits-valid-restriction",
                            "`totalDigits` weakens the base type's `totalDigits`",
                        ));
                    }
                    facets.total_digits = Some(count);
                } else {
                    if let Some(base_limit) = base_facets.fraction_digits
                        && count > base_limit
                    {
                        return Err(SchemaError::new(
                            "fractionDigits-valid-restriction",
                            "`fractionDigits` weakens the base type's `fractionDigits`",
                        ));
                    }
                    facets.fraction_digits = Some(count);
                }
            }
            "explicitTimezone" => {
                if !primitive.is_some_and(Primitive::has_timezone) {
                    return Err(SchemaError::new(
                        "cos-applicable-facets",
                        "`explicitTimezone` does not apply to this datatype",
                    ));
                }
                let rule = ExplicitTimezone::parse(&facet.value).ok_or_else(|| {
                    SchemaError::new(
                        "src-explicitTimezone",
                        format!("`{}` is not an `explicitTimezone` value", facet.value),
                    )
                })?;
                // Datatypes §4.3.13.3 (*explicitTimezone valid restriction*):
                // a base that *requires* or *prohibits* a timezone has already
                // fixed the question, so a restriction may only repeat its
                // answer (`saxonData/Zone/zone004.bad`, `zone005.bad`,
                // `oracleData/Zone/zone408.bad`, `zone409.bad`).
                if let Some(base) = facets.explicit_timezone
                    && base != ExplicitTimezone::Optional
                    && base != rule
                {
                    return Err(SchemaError::new(
                        "explicitTimezone-valid-restriction",
                        "`explicitTimezone` may not be relaxed by a restriction",
                    ));
                }
                facets.explicit_timezone = Some(rule);
            }
            "assertion" => {
                let assertion = crate::components::Assertion {
                    test: facet.value.clone(),
                    namespaces: facet.namespaces.clone(),
                    xpath_default_namespace: facet.xpath_default_namespace.clone(),
                };
                crate::assertion::check_static(&assertion, true, "src-assertion")?;
                facets.assertions.push(assertion);
            }
            other => {
                return Err(SchemaError::new(
                    "src-facet",
                    format!("`{other}` is not a constraining facet"),
                ));
            }
        }
    }

    // Datatypes §4.1.3 / Structures §3.16.6.4: a facet the base type fixed may
    // be restated, but not changed. `whiteSpace` has already been checked
    // above, with its own diagnostic.
    for name in overridden_fixed {
        if facet_value_differs(name, base_facets, &facets) {
            return Err(SchemaError::new(
                format!("{name}-valid-restriction"),
                format!("`{name}` is fixed by the base type, so its value may not differ"),
            ));
        }
    }

    if !step_patterns.is_empty() {
        facets.patterns.push(step_patterns);
        facets.pattern_sources.push(step_pattern_sources);
    }
    if !step_enumeration.is_empty() {
        facets.enumeration = Some(step_enumeration);
        facets.enumeration_lexical = step_enumeration_lexical;
    }

    if let (Some(min), Some(max)) = (facets.min_length, facets.max_length)
        && min > max
    {
        return Err(SchemaError::new(
            "minLength-less-than-equal-to-maxLength",
            "`minLength` is greater than `maxLength`",
        ));
    }
    if let (Some(total), Some(fraction)) = (facets.total_digits, facets.fraction_digits)
        && fraction > total
    {
        return Err(SchemaError::new(
            "fractionDigits-totalDigits",
            "`fractionDigits` is greater than `totalDigits`",
        ));
    }
    check_bounds_consistency(&facets)?;

    Ok(facets)
}

/// The facet pairs that may not both be written by one restriction step.
///
/// [XML Schema Part 2][p2] states these as schema component constraints on
/// `{facets}`:
///
/// * *length and minLength or maxLength* (§4.3.1.4, as amended by erratum
///   E2-27) — `length` may not be written beside `minLength` or `maxLength`
///   *unless* the bound in question is already a member of the base type's
///   `{facets}` and the written `length` lies inside it, which is what makes
///   `<xs:restriction base="xs:IDREFS"><xs:length value="5"/>
///   <xs:minLength value="1"/>` legal — `xs:IDREFS` carries `minLength="1"`
///   itself — while the same pair on `xs:string` is not;
/// * *maxInclusive and maxExclusive* (§4.3.9.4); and
/// * *minInclusive and minExclusive* (§4.3.10.4).
///
/// The check is per *step* — a derived type may perfectly well narrow a base
/// type's `maxInclusive` with a `maxExclusive`, which is why it cannot be
/// expressed on the merged [`Facets`].
///
/// [p2]: https://www.w3.org/TR/xmlschema-2/#rf-length
fn check_facet_cooccurrence(raw: &[RawFacet], base_facets: &Facets) -> Result<(), SchemaError> {
    let has = |name: &str| raw.iter().any(|facet| facet.name == name);
    let written = |name: &str| {
        raw.iter()
            .find(|facet| facet.name == name)
            .and_then(|facet| crate::value::collapse(&facet.value).parse::<usize>().ok())
    };
    if has("length") {
        let length = written("length");
        let inherited_ok =
            |bound: Option<usize>, base: Option<usize>, lower: bool| match (bound, base, length) {
                (Some(bound), Some(base_bound), Some(length)) => {
                    bound == base_bound
                        && if lower {
                            bound <= length
                        } else {
                            length <= bound
                        }
                }
                _ => false,
            };
        if has("minLength") && !inherited_ok(written("minLength"), base_facets.min_length, true) {
            return Err(SchemaError::new(
                "length-minLength-maxLength",
                "`length` may not appear beside a newly written `minLength` in one restriction",
            ));
        }
        if has("maxLength") && !inherited_ok(written("maxLength"), base_facets.max_length, false) {
            return Err(SchemaError::new(
                "length-minLength-maxLength",
                "`length` may not appear beside a newly written `maxLength` in one restriction",
            ));
        }
    }
    if has("maxInclusive") && has("maxExclusive") {
        return Err(SchemaError::new(
            "maxInclusive-maxExclusive",
            "`maxInclusive` and `maxExclusive` may not both appear in one restriction",
        ));
    }
    if has("minInclusive") && has("minExclusive") {
        return Err(SchemaError::new(
            "minInclusive-minExclusive",
            "`minInclusive` and `minExclusive` may not both appear in one restriction",
        ));
    }
    Ok(())
}

fn check_bounds_consistency(facets: &Facets) -> Result<(), SchemaError> {
    let lower = facets
        .min_inclusive
        .as_ref()
        .or(facets.min_exclusive.as_ref());
    let upper = facets
        .max_inclusive
        .as_ref()
        .or(facets.max_exclusive.as_ref());
    if let (Some(lower), Some(upper)) = (lower, upper)
        && let Some(ordering) = lower.partial_cmp_value(upper)
    {
        let strict = facets.min_exclusive.is_some() || facets.max_exclusive.is_some();
        let bad = match ordering {
            Ordering::Greater => true,
            Ordering::Equal => strict,
            Ordering::Less => false,
        };
        if bad {
            return Err(SchemaError::new(
                "minInclusive-less-than-equal-to-maxInclusive",
                "the lower bound is greater than the upper bound",
            ));
        }
    }
    Ok(())
}

fn parse_count(name: &str, value: &str) -> Result<usize, SchemaError> {
    crate::value::collapse(value).parse::<usize>().map_err(|_| {
        SchemaError::new(
            format!("src-{name}"),
            format!("`{value}` is not a non-negative integer"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SchemaSet;

    fn builtin_schema() -> Schema {
        SchemaSet::new().compile().expect("the built-in schema")
    }

    fn check(schema: &Schema, local: &str, lexical: &str) -> bool {
        let Some(id) = schema.builtin(local) else {
            return false;
        };
        validate(schema, id, lexical, &NamespaceContext::new()).is_ok()
    }

    #[test]
    fn integer_bounds_come_from_the_builtin_facets() {
        let schema = builtin_schema();
        assert!(check(&schema, "byte", "127"));
        assert!(!check(&schema, "byte", "128"));
        assert!(check(&schema, "unsignedLong", "18446744073709551615"));
        assert!(!check(&schema, "unsignedLong", "18446744073709551616"));
        assert!(check(&schema, "positiveInteger", "1"));
        assert!(!check(&schema, "positiveInteger", "0"));
        assert!(!check(&schema, "integer", "1.0"));
    }

    #[test]
    fn whitespace_is_collapsed_for_all_but_string() {
        let schema = builtin_schema();
        assert!(check(&schema, "integer", "  12\n"));
        assert!(check(&schema, "string", "  12\n"));
        assert!(!check(&schema, "NCName", "a b"));
        assert!(check(&schema, "token", " a  b "));
    }

    #[test]
    fn list_builtins_split_on_whitespace() {
        let schema = builtin_schema();
        assert!(check(&schema, "NMTOKENS", "a b c"));
        assert!(!check(&schema, "NMTOKENS", ""));
        assert!(!check(&schema, "IDREFS", "1a"));
    }

    #[test]
    fn name_patterns_are_enforced() {
        let schema = builtin_schema();
        assert!(check(&schema, "Name", "a:b"));
        assert!(!check(&schema, "NCName", "a:b"));
        assert!(check(&schema, "language", "en-GB"));
        assert!(!check(&schema, "language", "-en"));
    }

    /// The `xs:duration` seconds component needs a digit on each side of its
    /// decimal point, even though `xs:decimal` itself accepts `12.` and `.5`.
    /// XSTS `saxonData/Simple/simple086` states the rule against a bare
    /// `<xs:element name="elem" type="xs:duration"/>`, so this exercises it
    /// where that test does: through the built-in type's own lexical mapping,
    /// not through a facet.
    #[test]
    fn duration_seconds_fraction_needs_digits_on_both_sides() {
        let schema = builtin_schema();
        // simple086.v01.xml
        assert!(check(&schema, "duration", "PT12H30M12.5S"));
        // simple086.n01.xml ("trailing '.' not allowed").
        assert!(!check(&schema, "duration", "PT12H30M12.S"));
        // simple086.n02.xml ("leading '.' not allowed").
        assert!(!check(&schema, "duration", "PT12H30M.5S"));
        // The same lexical space backs `xs:dayTimeDuration` in 1.1.
        let one_one = SchemaSet::new()
            .with_version(crate::XsdVersion::V1_1)
            .compile()
            .expect("the built-in schema");
        assert!(check(&one_one, "dayTimeDuration", "PT10M30.5S"));
        assert!(!check(&one_one, "dayTimeDuration", "PT10M30.S"));
        assert!(!check(&one_one, "dayTimeDuration", "PT10M.5S"));
    }

    /// A `pattern` facet is read against the *schema's* version, not against
    /// one fixed grammar: XSD 1.0's Appendix F confines a bare `-` to the
    /// first or last member of a character group, and XSD 1.1's Appendix G
    /// does not. The XML Schema Test Suite pins both directions on one
    /// pattern — `msData/regex/reH19`-`reH21` (`[a-a-x-x]+`) carry both
    /// `<expected version="1.0" validity="invalid"/>` and `<expected
    /// version="1.1" validity="valid"/>` — so a `src-pattern` error here for
    /// 1.0 and a clean compile for 1.1 is what conformance requires. Driven
    /// through a whole schema document rather than through `apply_facets`
    /// directly, so it also covers the version reaching the facet at all.
    #[test]
    fn pattern_facet_grammar_follows_the_schema_version() {
        const DOCUMENT: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="e">
                <xs:simpleType>
                  <xs:restriction base="xs:string">
                    <xs:pattern value="[a-a-x-x]+"/>
                  </xs:restriction>
                </xs:simpleType>
              </xs:element>
            </xs:schema>"#;

        for (version, expected_valid) in [
            (crate::XsdVersion::V1_0, false),
            (crate::XsdVersion::V1_1, true),
        ] {
            let mut set = SchemaSet::new().with_version(version);
            set.add_document(None, DOCUMENT)
                .expect("the document itself is well-formed");
            let compiled = set.compile();
            assert_eq!(
                compiled.is_ok(),
                expected_valid,
                "reH19-reH21 under {version:?}: {compiled:?}"
            );
            if let Err(error) = compiled {
                assert_eq!(error.code(), "src-pattern");
            }
        }
    }

    /// Datatypes §4.3.6: a union borrows its normalization from whichever
    /// member type accepts the value, so an `xs:normalizedString` member keeps
    /// a trailing space that the union's own `collapse` would have dropped.
    /// `msData/simpleType/stE063` turns on exactly that: its `root` is a union
    /// over `xs:gYearMonth xsd:normalizedString restrictedNotation` with
    /// `fixed="abcd edfgh "`, and the instance `<root>abcd edfgh</root>` is
    /// *invalid* — which it can only be if the two normalizations differ.
    #[test]
    fn a_union_normalizes_with_the_member_that_accepted_the_value() {
        const DOCUMENT: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:simpleType name="u">
                <xs:union memberTypes="xs:gYearMonth xs:normalizedString"/>
              </xs:simpleType>
              <xs:simpleType name="n">
                <xs:union memberTypes="xs:gYearMonth xs:int"/>
              </xs:simpleType>
            </xs:schema>"#;

        let mut set = SchemaSet::new();
        set.add_document(None, DOCUMENT)
            .expect("the document itself is well-formed");
        let schema = set.compile().expect("the unions compile");
        let value = |local: &str, lexical: &str| {
            let id = schema.find_type(None, local).expect("the named union");
            validate(&schema, id, lexical, &NamespaceContext::new())
        };

        let fixed = value("u", "abcd edfgh ").expect("`normalizedString` accepts it");
        assert_eq!(fixed.normalized, "abcd edfgh ");
        assert_eq!(fixed.value, Value::String("abcd edfgh ".to_owned()));
        let instance = value("u", "abcd edfgh").expect("`normalizedString` accepts it too");
        assert_eq!(instance.normalized, "abcd edfgh");
        assert_ne!(
            fixed.value, instance.value,
            "the trailing space distinguishes the two values"
        );

        // `normalizedString` is `whiteSpace=\"replace\"`, so a tab becomes a
        // space and is *not* collapsed away with the one beside it.
        assert_eq!(
            value("u", "a\t b")
                .expect("`normalizedString` accepts it")
                .normalized,
            "a  b"
        );

        // A member that does collapse still collapses: the union defers to the
        // member, it does not stop normalizing.
        let integer = value("n", "  12\n").expect("`xs:int` accepts it");
        assert_eq!(integer.normalized, "12");
        assert!(value("n", "1 2").is_err());
    }

    /// The negative half of the rule above: XSD 1.0 restricts *where* a bare
    /// `-` may stand inside a character group, it does not outlaw it, so a
    /// pattern the suite accepts unconditionally must still compile under
    /// both versions. `xs:language`'s own built-in pattern is one such (it
    /// ends `(-[a-zA-Z0-9]{1,8})*`), and is already exercised for 1.0 by
    /// [`name_patterns_are_enforced`]; `[a-z-]` is the shape the version
    /// split comes closest to touching.
    #[test]
    fn trailing_hyphen_pattern_compiles_under_both_versions() {
        const DOCUMENT: &str = r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
              <xs:element name="e">
                <xs:simpleType>
                  <xs:restriction base="xs:string">
                    <xs:pattern value="[a-z-]+"/>
                  </xs:restriction>
                </xs:simpleType>
              </xs:element>
            </xs:schema>"#;

        for version in [crate::XsdVersion::V1_0, crate::XsdVersion::V1_1] {
            let mut set = SchemaSet::new().with_version(version);
            set.add_document(None, DOCUMENT)
                .expect("the document itself is well-formed");
            assert!(
                set.compile().is_ok(),
                "`[a-z-]+` must stay valid under {version:?}"
            );
        }
    }
}
