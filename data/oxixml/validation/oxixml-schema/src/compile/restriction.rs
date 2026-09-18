//! *Particle Valid (Restriction)* and the attribute half of *Derivation Valid
//! (Restriction, Complex)* — [§3.9.6][pvr] and [§3.4.6][dvr].
//!
//! When a complex type restricts another, its content model must accept only
//! sequences the base already accepted. The Recommendation states this as a
//! recursion over the two particle trees with one case per pair of terms
//! (`NameAndTypeOK`, `NSCompat`, `NSSubset`, `Recurse`, `RecurseLax`,
//! `RecurseUnordered`, `MapAndSum`, `NSRecurseCheckCardinality`), and that is
//! what this module implements.
//!
//! Where a case is genuinely undecidable without more machinery than the
//! constraint is worth — an occurrence range that cannot be summed because a
//! member is unbounded, say — the check *accepts*, so a legal schema is never
//! rejected by an approximation.
//!
//! [pvr]: https://www.w3.org/TR/xmlschema-1/#rcase-RecurseLax
//! [dvr]: https://www.w3.org/TR/xmlschema-1/#derivation-ok-restriction

use crate::components::{
    AttributeUseKind, Compositor, ContentType, Derivation, ElementId, MaxOccurs, ModelGroup,
    Particle, ProcessContents, Term, TypeId, TypeKind, Wildcard,
};
use crate::error::SchemaError;
use crate::schema::Schema;

use super::Compiler;

impl Compiler {
    /// Checks every complex type that derives by restriction.
    pub(crate) fn check_restrictions(&self) -> Result<(), SchemaError> {
        for index in 0..self.schema.types.len() {
            let id = TypeId(u32::try_from(index).unwrap_or(u32::MAX));
            let TypeKind::Complex(complex) = &self.schema.types[index].kind else {
                continue;
            };
            let Some(base) = self.schema.types[index].base else {
                continue;
            };
            if base == id || base == self.schema.any_type {
                continue;
            }
            let Some(base_complex) = self.schema.types[base.index()].as_complex() else {
                continue;
            };
            if complex.derivation != Derivation::Restriction {
                if self.schema.version().is_1_1() {
                    let what = format!(
                        "`{}` is not a valid extension of `{}`",
                        self.schema.type_name(id),
                        self.schema.type_name(base)
                    );
                    check_open_content_extension(complex, base_complex, &what)?;
                }
                continue;
            }
            let what = format!(
                "`{}` is not a valid restriction of `{}`",
                self.schema.type_name(id),
                self.schema.type_name(base)
            );
            check_attribute_uses(&self.schema, complex, base_complex, &what)?;
            if self.schema.version().is_1_1() {
                check_open_content_restriction(complex, base_complex, &what)?;
            }
            check_content(&self.schema, &complex.content, &base_complex.content, &what)?;
        }
        Ok(())
    }

    /// §4.2.2 *src-redefine.6.2.2* and *7.2.2*: a model group or attribute
    /// group redefinition that does **not** refer to itself replaces the
    /// original outright, and must therefore be a valid restriction of it
    /// (`msData/schema/schL1`, `msData/schema/schM3`).
    pub(crate) fn check_redefine_restrictions(&self) -> Result<(), SchemaError> {
        for (kind, real, hidden) in &self.redefine_restrictions {
            let what = format!("the redefinition of `{real}` does not restrict the original");
            match kind {
                super::RedefineKind::Group => {
                    let (Some(new_id), Some(old_id)) = (
                        self.schema.global_groups.get(real).copied(),
                        self.schema.global_groups.get(hidden).copied(),
                    ) else {
                        continue;
                    };
                    let (Some(new_group), Some(old_group)) = (
                        self.schema.groups[new_id.index()].model_group.clone(),
                        self.schema.groups[old_id.index()].model_group.clone(),
                    ) else {
                        continue;
                    };
                    let wrap = |group: ModelGroup| Particle {
                        min_occurs: 1,
                        max_occurs: MaxOccurs::Bounded(1),
                        term: Term::Group(group),
                    };
                    particle_restricts(&self.schema, &wrap(new_group), &wrap(old_group), &what)?;
                }
                super::RedefineKind::AttributeGroup => {
                    let (Some(new_id), Some(old_id)) = (
                        self.schema.global_attribute_groups.get(real).copied(),
                        self.schema.global_attribute_groups.get(hidden).copied(),
                    ) else {
                        continue;
                    };
                    attribute_uses_restrict(
                        &self.schema,
                        &self.schema.attribute_groups[new_id.index()],
                        &self.schema.attribute_groups[old_id.index()],
                        &what,
                    )?;
                }
                super::RedefineKind::Type => {}
            }
        }
        Ok(())
    }
}

/// XSD 1.1 §3.4.6 *Derivation Valid (Restriction, Complex)*, the open-content
/// clause: a restriction may only narrow the base's open content, so it must
/// have one at all, may not turn a `suffix` base into an `interleave`, and its
/// wildcard must be a ·subset· of the base's
/// (`saxonData/Open/open016.bad` … `open019.bad`).
fn check_open_content_restriction(
    derived: &crate::components::ComplexTypeInfo,
    base: &crate::components::ComplexTypeInfo,
    what: &str,
) -> Result<(), SchemaError> {
    let Some(own) = derived.open_content.as_ref().filter(open_applies) else {
        return Ok(());
    };
    // When the restriction's own content model accepts nothing but the empty
    // sequence, `interleave` and `suffix` describe the same language, and every
    // element the open content admits may equally have been admitted by the
    // base's *particle* rather than by open content of its own — a case this
    // implementation does not decide and therefore accepts
    // (`saxonData/Open/open020`, `open021`, `open022`).
    if matches_nothing(&derived.content) {
        return Ok(());
    }
    let Some(theirs) = base.open_content.as_ref().filter(open_applies) else {
        return Err(SchemaError::new(
            "derivation-ok-restriction.5.4.2",
            format!("{what}: the base type has no open content"),
        ));
    };
    if own.interleave && !theirs.interleave {
        return Err(SchemaError::new(
            "derivation-ok-restriction.5.4.2",
            format!("{what}: the base type's open content is `suffix`, not `interleave`"),
        ));
    }
    open_wildcard_subset(theirs, own, what, "derivation-ok-restriction.5.4.2")
}

/// XSD 1.1 §3.4.6 *Derivation Valid (Extension)*: an `interleave` base stays
/// `interleave` (`saxonData/Open/open030.bad`, `open033.bad`).
///
/// The two *wildcards* are **united**, not nested, so nothing requires one to
/// be a subset of the other: `saxonData/Open/open047` extends open content in
/// one namespace with open content in every other namespace and is valid.
fn check_open_content_extension(
    derived: &crate::components::ComplexTypeInfo,
    base: &crate::components::ComplexTypeInfo,
    what: &str,
) -> Result<(), SchemaError> {
    let Some(theirs) = base.open_content.as_ref().filter(open_applies) else {
        return Ok(());
    };
    let Some(own) = derived.open_content.as_ref().filter(open_applies) else {
        // The extension wrote none of its own, so it inherits the base's
        // (`saxonData/Open/open027`; `open031` writes `mode="none"` and is
        // valid too).
        return Ok(());
    };
    if theirs.interleave && !own.interleave {
        return Err(SchemaError::new(
            "cos-ct-extends.1.4.3.3",
            format!("{what}: the base type's open content is `interleave`, not `suffix`"),
        ));
    }
    Ok(())
}

/// Whether a content type accepts no element children at all.
fn matches_nothing(content: &ContentType) -> bool {
    match content.particle() {
        None => true,
        Some(particle) => particle_matches_nothing(particle),
    }
}

fn particle_matches_nothing(particle: &Particle) -> bool {
    match &particle.term {
        Term::Element(_) | Term::Wildcard(_) => particle.max_occurs == MaxOccurs::Bounded(0),
        Term::Group(group) => group.particles.iter().all(particle_matches_nothing),
    }
}

/// Whether an `xs:openContent` contributes a wildcard at all (`mode="none"`
/// does not).
fn open_applies(open: &&crate::components::OpenContent) -> bool {
    open.wildcard.is_some()
}

/// *Wildcard Subset* (§3.10.6) between two open-content wildcards.
fn open_wildcard_subset(
    outer: &crate::components::OpenContent,
    inner: &crate::components::OpenContent,
    what: &str,
    code: &str,
) -> Result<(), SchemaError> {
    let (Some(outer), Some(inner)) = (outer.wildcard.as_ref(), inner.wildcard.as_ref()) else {
        return Ok(());
    };
    if !outer.namespace.is_superset_of(&inner.namespace)
        || process_rank(inner.process_contents) < process_rank(outer.process_contents)
        || !super::subsume::covers_disallowed(outer, inner)
    {
        return Err(SchemaError::new(
            code,
            format!("{what}: the open-content wildcards are not in a subset relation"),
        ));
    }
    Ok(())
}

fn check_attribute_uses(
    schema: &Schema,
    derived: &crate::components::ComplexTypeInfo,
    base: &crate::components::ComplexTypeInfo,
    what: &str,
) -> Result<(), SchemaError> {
    for use_ in &derived.attribute_uses {
        let declaration = schema.attribute_declaration(use_.attribute);
        let matching = base.attribute_uses.iter().find(|candidate| {
            schema.attribute_declaration(candidate.attribute).name == declaration.name
        });
        match matching {
            Some(base_use) => {
                if base_use.kind == AttributeUseKind::Required
                    && use_.kind != AttributeUseKind::Required
                {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.2.1.1",
                        format!(
                            "{what}: the attribute `{}` is required in the base type",
                            declaration.name
                        ),
                    ));
                }
                // XSD 1.1 §3.4.6 clause 2.1.2: an attribute use may not change
                // whether the attribute is *inheritable*, because conditional
                // type assignment elsewhere in the document depends on it
                // (`saxonData/CTA/cta9004err`, `cta9005err`).
                if schema.version().is_1_1()
                    && schema.attribute_declaration(base_use.attribute).inheritable
                        != declaration.inheritable
                {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.2.1.2",
                        format!(
                            "{what}: the attribute `{}` changes `inheritable`",
                            declaration.name
                        ),
                    ));
                }
                let base_type = schema.attribute_declaration(base_use.attribute).type_id;
                if base_type != declaration.type_id
                    && !schema.is_derived_from(declaration.type_id, base_type)
                {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.2.1.2",
                        format!(
                            "{what}: the type of the attribute `{}` is not derived from its type \
                             in the base",
                            declaration.name
                        ),
                    ));
                }
                let base_constraint = base_use.value_constraint.as_ref().or(schema
                    .attribute_declaration(base_use.attribute)
                    .value_constraint
                    .as_ref());
                if let Some(base_constraint) = base_constraint
                    && base_constraint.fixed
                {
                    let own = use_
                        .value_constraint
                        .as_ref()
                        .or(declaration.value_constraint.as_ref());
                    let matches =
                        own.is_some_and(|own| own.fixed && own.value == base_constraint.value);
                    if !matches {
                        return Err(SchemaError::new(
                            "derivation-ok-restriction.2.1.3",
                            format!(
                                "{what}: the attribute `{}` is fixed in the base type",
                                declaration.name
                            ),
                        ));
                    }
                }
            }
            None => {
                // §3.4.6 clause 2.2: an attribute the base does not declare is
                // only allowed if the base's attribute wildcard admits it.
                let admitted = base.attribute_wildcard.as_ref().is_some_and(|wildcard| {
                    wildcard.namespace.allows(declaration.name.namespace())
                });
                if !admitted {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.2.2",
                        format!(
                            "{what}: the base type has no attribute `{}` and no wildcard that \
                             admits it",
                            declaration.name
                        ),
                    ));
                }
            }
        }
    }
    // The base's required attributes must survive the restriction.
    for base_use in &base.attribute_uses {
        if base_use.kind != AttributeUseKind::Required {
            continue;
        }
        let name = &schema.attribute_declaration(base_use.attribute).name;
        let kept = derived
            .attribute_uses
            .iter()
            .any(|candidate| &schema.attribute_declaration(candidate.attribute).name == name);
        if !kept {
            return Err(SchemaError::new(
                "derivation-ok-restriction.3",
                format!("{what}: the required attribute `{name}` is missing"),
            ));
        }
    }
    if let Some(wildcard) = &derived.attribute_wildcard {
        match &base.attribute_wildcard {
            Some(base_wildcard) => {
                // *Wildcard Subset* (§3.10.6) is about names, not only
                // namespaces: the restricting wildcard must also refuse
                // everything the base's `notQName` refuses
                // (`saxonData/Wild/wild057`).
                if !base_wildcard.namespace.is_superset_of(&wildcard.namespace)
                    || !super::subsume::covers_disallowed(base_wildcard, wildcard)
                {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.4.2",
                        format!(
                            "{what}: the attribute wildcard is not a subset of the base type's"
                        ),
                    ));
                }
                // XSD 1.0 Errata E1-21 added to clause 4 what the element half
                // of the constraint (*rcase-NSSubset.3*) already said: a
                // restriction may not relax how the attributes the wildcard
                // matches are assessed, so its `processContents` must be as
                // strong as the base's or stronger — `strict` over `lax` over
                // `skip` (`msData/errata10/errC009`).
                if process_rank(wildcard.process_contents)
                    < process_rank(base_wildcard.process_contents)
                {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.4.3",
                        format!(
                            "{what}: the attribute wildcard weakens the base type's \
                             `processContents`"
                        ),
                    ));
                }
            }
            None => {
                return Err(SchemaError::new(
                    "derivation-ok-restriction.4.1",
                    format!("{what}: the base type has no attribute wildcard"),
                ));
            }
        }
    }
    Ok(())
}

fn check_content(
    schema: &Schema,
    derived: &ContentType,
    base: &ContentType,
    what: &str,
) -> Result<(), SchemaError> {
    match (derived, base) {
        // Simple content restricting anything is checked where the simple type
        // itself is derived.
        (ContentType::Simple(_), _) => Ok(()),
        // §3.4.6 clause 5.2: an *empty* content type restricts a base whose
        // content type is empty as well, or one that is mixed with an
        // ·emptiable· particle. A base with **simple** content is neither: it
        // requires a value where the restriction admits nothing at all, so a
        // `<complexContent><restriction>` with no particle over a
        // `<simpleContent>` base is in error however the base's simple type
        // treats the empty string (`msData/particles/particlesZ039`).
        (ContentType::Empty, ContentType::Simple(_)) => Err(SchemaError::new(
            "derivation-ok-restriction.5.2.2",
            format!("{what}: the base type has simple content, which cannot become empty"),
        )),
        // §3.4.6 clause 5.3: an empty content type is a valid restriction of any
        // content type whose particle can match the empty sequence.
        (ContentType::Empty, _) => match base.particle() {
            Some(particle) if !particle.is_emptiable() => Err(SchemaError::new(
                "derivation-ok-restriction.5.3.2",
                format!("{what}: the base type's content model cannot be empty"),
            )),
            _ => Ok(()),
        },
        // §3.4.6 clause 5: when the base has *empty* content, the only clause
        // that can apply is 5.2, which needs the restriction's content type to
        // be empty as well. A particle that can match even one element is
        // therefore an error however emptiable it is
        // (`msData/particles/particlesJb006`).
        (_, ContentType::Empty) => match derived.particle() {
            Some(_) => Err(SchemaError::new(
                "derivation-ok-restriction.5.3.1",
                format!("{what}: the base type has empty content"),
            )),
            None => Ok(()),
        },
        (_, ContentType::Simple(_)) => Err(SchemaError::new(
            "derivation-ok-restriction.5.3.1",
            format!("{what}: the base type has no element content to restrict"),
        )),
        (ContentType::Mixed(_), ContentType::ElementOnly(_)) => Err(SchemaError::new(
            "derivation-ok-restriction.5.4.1.2",
            format!("{what}: mixed content may not restrict element-only content"),
        )),
        (derived, base) => {
            let (Some(derived), Some(base)) = (derived.particle(), base.particle()) else {
                return Ok(());
            };
            particle_valid_restriction(schema, derived, base, what, 0)
        }
    }
}

/// How deep the recursion may go before it simply accepts; content models this
/// deep do not occur outside adversarial input.
const DEPTH_LIMIT: usize = 64;

fn particle_valid_restriction(
    schema: &Schema,
    derived: &Particle,
    base: &Particle,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    if depth > DEPTH_LIMIT {
        return Ok(());
    }
    let derived = &simplify(derived);
    let base = &simplify(base);
    // XSD 1.1 §3.9.6 replaced the 1.0 case table with a statement about
    // languages, which for a base `all` group can be decided exactly; see
    // `super::subsume`.
    if super::subsume::applies(schema, base) {
        return super::subsume::all_restriction(schema, derived, base, what);
    }
    match (&derived.term, &base.term) {
        (Term::Element(r), Term::Element(b)) => {
            name_and_type_ok(schema, derived, *r, base, *b, what)
        }
        (Term::Group(r), Term::Element(base_element)) => {
            restrict_substitution_group(schema, derived, r, base, *base_element, what, depth)
        }
        (_, Term::Element(_)) => Err(SchemaError::new(
            "rcase-Forbidden",
            format!("{what}: an element particle cannot be restricted by a group or wildcard"),
        )),
        (Term::Element(element), Term::Wildcard(wildcard)) => {
            ns_compat(schema, derived, *element, base, wildcard, what)
        }
        (Term::Wildcard(r), Term::Wildcard(b)) => ns_subset(schema, derived, r, base, b, what),
        (Term::Group(group), Term::Wildcard(wildcard)) => {
            ns_recurse_check_cardinality(schema, derived, group, base, wildcard, what, depth)
        }
        (Term::Wildcard(_), Term::Group(_)) => Err(SchemaError::new(
            "rcase-Forbidden",
            format!("{what}: a model group cannot be restricted by a wildcard"),
        )),
        (Term::Element(element), Term::Group(group)) => {
            recurse_as_if_group(schema, derived, *element, base, group, what, depth)
        }
        (Term::Group(r), Term::Group(b)) => {
            group_restriction(schema, derived, r, base, b, what, depth)
        }
    }
}

/// A model group restricting an element particle whose element heads a
/// ·substitution group·.
///
/// The table in §3.9.6 has no cell for a group restricting an element and calls
/// it forbidden, which is right when the base particle can match exactly one
/// name. A particle referring to the head of a substitution group cannot: it
/// admits the head *and* every member, so it already denotes a choice. A
/// restriction that offers some of those members is narrowing that choice, and
/// accepts strictly fewer sequences than the base does
/// (`msData/element/elemZ027a`, where a `choice` of two members restricts a
/// particle referring to their head).
///
/// The base is therefore rewritten as the choice it stands for — the head
/// first, then the members that can actually stand in for it — and the ordinary
/// group-against-group cases decide the rest. `elemZ027c` is the control: a
/// head part-way up its chain carries `block="substitution"`, so the member the
/// restriction offers is not in the ·actual· substitution group, is not among
/// the branches, and the restriction is rejected exactly as before.
fn restrict_substitution_group(
    schema: &Schema,
    derived: &Particle,
    r: &ModelGroup,
    base: &Particle,
    base_element: ElementId,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    let members: Vec<ElementId> = schema
        .element_declaration(base_element)
        .substitutable_by
        .iter()
        .copied()
        .filter(|member| substitutes_for(schema, *member, base_element))
        .collect();
    if members.is_empty() {
        return group_restricts_element(schema, derived, r, base, base_element, what, depth);
    }
    let branches = std::iter::once(base_element)
        .chain(members)
        .map(|id| Particle {
            min_occurs: 1,
            max_occurs: MaxOccurs::Bounded(1),
            term: Term::Element(id),
        })
        .collect();
    let expanded = ModelGroup {
        compositor: Compositor::Choice,
        particles: branches,
    };
    let expanded_base = Particle {
        min_occurs: base.min_occurs,
        max_occurs: base.max_occurs,
        term: Term::Group(expanded.clone()),
    };
    group_restriction(schema, derived, r, &expanded_base, &expanded, what, depth)
}

/// A model group restricting an element particle whose element heads no
/// ·substitution group·, under XSD 1.1's language reading of §3.9.6.
///
/// The base particle admits exactly one name, repeated within its occurrence
/// range, so a group restricts it precisely when
///
/// * every element information item the group can produce is governed by a
///   declaration that is itself a ·valid restriction· of the base's — which,
///   the base admitting one name only, means every leaf of the group must be
///   an element particle passing `NameAndTypeOK`; and
/// * the number of items the group can produce — its ·effective total range·
///   (§3.8.6) — lies within the base particle's occurrence range.
///
/// The table in §3.9.6 has no cell for this pair and calls it forbidden, which
/// is why the suite marks `msData/particles/particlesHb011` invalid for 1.0 and
/// valid for 1.1: there a `<sequence maxOccurs="2">` holding one
/// `<element name="e1" maxOccurs="2"/>` restricts the base's
/// `<element name="e1" minOccurs="0" maxOccurs="10"/>`, contributing between
/// one and four `e1`s where the base admits up to ten.
fn group_restricts_element(
    schema: &Schema,
    derived: &Particle,
    group: &ModelGroup,
    base: &Particle,
    base_element: ElementId,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    let forbidden = || {
        SchemaError::new(
            "rcase-Forbidden",
            format!("{what}: an element particle cannot be restricted by a group or wildcard"),
        )
    };
    if !schema.version().is_1_1() {
        return Err(forbidden());
    }
    let (min, max) = effective_total_range(derived, group, depth);
    if min < base.min_occurs || !fits_within(max, base.max_occurs) {
        return Err(SchemaError::new(
            "rcase-Forbidden",
            format!(
                "{what}: the group may contribute more of `{}` than the base type admits",
                schema.element_declaration(base_element).name
            ),
        ));
    }
    let once = |term: &Term| Particle {
        min_occurs: 1,
        max_occurs: MaxOccurs::Bounded(1),
        term: term.clone(),
    };
    let mut leaves = Vec::new();
    if collect_leaves(group, depth, &mut leaves).is_err() {
        return Err(forbidden());
    }
    for leaf in leaves {
        particle_valid_restriction(schema, &once(&leaf), &once(&base.term), what, depth + 1)?;
    }
    Ok(())
}

/// Collects the terms of every element or wildcard particle inside `group`.
///
/// Fails past [`DEPTH_LIMIT`] levels of nesting rather than recursing without a
/// bound; the caller then falls back on the §3.9.6 table's own answer.
fn collect_leaves(group: &ModelGroup, depth: usize, out: &mut Vec<Term>) -> Result<(), ()> {
    if depth > DEPTH_LIMIT {
        return Err(());
    }
    for particle in &group.particles {
        match &particle.term {
            Term::Group(inner) => collect_leaves(inner, depth + 1, out)?,
            other => out.push(other.clone()),
        }
    }
    Ok(())
}

/// Whether `value` is within `limit`, both being maximum occurrence counts.
fn fits_within(value: MaxOccurs, limit: MaxOccurs) -> bool {
    match (value, limit) {
        (_, MaxOccurs::Unbounded) => true,
        (MaxOccurs::Unbounded, MaxOccurs::Bounded(_)) => false,
        (MaxOccurs::Bounded(value), MaxOccurs::Bounded(limit)) => value <= limit,
    }
}

/// `RecurseAsIfGroup` (§3.9.6): an element particle restricting a group
/// particle is decided by treating the element as a group of the base's own
/// compositor holding just that element, and re-asking the group-against-group
/// question. The wrapper is handed straight to [`group_restriction`] rather
/// than back through [`particle_valid_restriction`], which would strip it off
/// again as a pointless particle and loop.
///
/// The rule leaves *where the occurrence range goes* open, and the two readings
/// are not equivalent:
///
/// * on the **inner** member, with the fake group occurring exactly once — the
///   element's whole range matched against one of the base's particles. This is
///   what the XSD 1.0 wording says, and a 1.0 processor is held to it:
///   `msData/particles/particlesHa161` is marked invalid for 1.0 and valid for
///   1.1 outright, and `particlesK006` invalid for both.
/// * on the **outer** fake group particle, with the single member pinned to
///   exactly one — one activation of the element matched against one of the
///   base's own particles, the whole thing repeated as often as the element may
///   occur. This is XSD 1.1's clarification.
///
/// The suite needs *both* under 1.1: `particlesHa161` and
/// `Particles_w3c/particlesZ001` need the second, while `particlesM002`,
/// `particlesM003`, `particlesM035`, `particlesW008`, `particlesHb010`,
/// `particlesIb001`, `ComplexType_w3c/ctF007` and `Additional_w3c/addA005` are
/// elements whose range exceeds the base *group*'s while fitting the base
/// *member*'s, which only the first accepts — and every one of those is marked
/// valid for both versions. So under 1.1 either reading suffices, which is what
/// the rest of this module already does wherever a case is undecidable: both
/// readings are strictly stronger than the language containment §3.9.6 is
/// really about, so accepting under either still rejects nothing the
/// Recommendation admits.
fn recurse_as_if_group(
    schema: &Schema,
    derived: &Particle,
    element: ElementId,
    base: &Particle,
    group: &ModelGroup,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    const ONCE: (u32, MaxOccurs) = (1, MaxOccurs::Bounded(1));
    let own = (derived.min_occurs, derived.max_occurs);
    let mut readings = vec![(own, ONCE)];
    if schema.version().is_1_1() {
        readings.push((ONCE, own));
    }
    let mut failure = None;
    for (inner_occurs, outer_occurs) in readings {
        let wrapper = ModelGroup {
            compositor: group.compositor,
            particles: vec![Particle {
                min_occurs: inner_occurs.0,
                max_occurs: inner_occurs.1,
                term: Term::Element(element),
            }],
        };
        let wrapped = Particle {
            min_occurs: outer_occurs.0,
            max_occurs: outer_occurs.1,
            term: Term::Group(wrapper.clone()),
        };
        match group_restriction(schema, &wrapped, &wrapper, base, group, what, depth) {
            Ok(()) => return Ok(()),
            Err(error) => {
                // The first reading's diagnostic is the one reported: it is the
                // one the Recommendation states in its own words.
                failure.get_or_insert(error);
            }
        }
        if own == ONCE {
            // The two readings coincide; there is nothing else to try.
            break;
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The four group-against-group cases of the table in §3.9.6.
fn group_restriction(
    schema: &Schema,
    derived: &Particle,
    r: &ModelGroup,
    base: &Particle,
    b: &ModelGroup,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    match (r.compositor, b.compositor) {
        (Compositor::All, Compositor::All) | (Compositor::Sequence, Compositor::Sequence) => {
            occurrence_ok(schema, derived, base, what)?;
            let r_particles = flatten(r.compositor, &r.particles);
            let b_particles = flatten(b.compositor, &b.particles);
            recurse(schema, &r_particles, &b_particles, what, depth, true)
        }
        (Compositor::Choice, Compositor::Choice) => {
            occurrence_ok(schema, derived, base, what)?;
            let r_particles = flatten(r.compositor, &r.particles);
            let b_particles = flatten(b.compositor, &b.particles);
            if schema.version().is_1_1() {
                recurse_lax_unordered(schema, &r_particles, &b_particles, what, depth)
            } else {
                recurse(schema, &r_particles, &b_particles, what, depth, false)
            }
        }
        (Compositor::Sequence, Compositor::All) => {
            occurrence_ok(schema, derived, base, what)?;
            let r_particles = flatten(r.compositor, &r.particles);
            let b_particles = flatten(b.compositor, &b.particles);
            recurse_unordered(schema, &r_particles, &b_particles, what, depth)
        }
        (Compositor::Sequence, Compositor::Choice) => {
            map_and_sum(schema, derived, r, base, b, what, depth)
        }
        _ => Err(SchemaError::new(
            "rcase-Forbidden",
            format!(
                "{what}: a `{}` group cannot restrict a `{}` group",
                compositor_name(r.compositor),
                compositor_name(b.compositor)
            ),
        )),
    }
}

/// Removes the *pointless* group particles of §3.9.6: a model group that
/// occurs exactly once and holds exactly one particle contributes nothing of
/// its own, so it stands for that particle.
///
/// Without this, `<xs:sequence><xs:any minOccurs="3" maxOccurs="3"/></xs:sequence>`
/// would be a `sequence` rather than the wildcard it denotes, and restricting
/// it by an `xs:all` group would land in the "forbidden" cell of the table in
/// §3.9.6 instead of `NSRecurseCheckCardinality`
/// (`msData/particles/particlesHa070`).
fn simplify(particle: &Particle) -> Particle {
    let mut current = particle.clone();
    loop {
        let Term::Group(group) = &current.term else {
            return current;
        };
        if current.min_occurs != 1 || current.max_occurs != MaxOccurs::Bounded(1) {
            return current;
        }
        let [only] = group.particles.as_slice() else {
            return current;
        };
        current = only.clone();
    }
}

/// Splices out *pointless nested groups* of the same compositor as the list
/// they sit in, complementing [`simplify`]'s single-particle case: a group
/// particle that occurs exactly once and shares its containing list's own
/// compositor describes exactly the same language whether it is written as a
/// nested group or with its particles spliced directly into the parent —
/// sequence and choice concatenation are both associative.
///
/// `<xs:group ref="…"/>` is the most common source of such particles — the
/// component model gives a group reference no wrapping of its own beyond the
/// referenced group's model group verbatim — so without this, a base type
/// that merely factors part of a `sequence` out into a named group can no
/// longer be validly restricted at the point the factored part begins: the
/// restriction's own (already-inlined) particles line up one-for-one against
/// the base's, but the base's group-reference particle presents itself as a
/// single opaque slot standing for *all* of the factored group's particles
/// at once, so `Recurse`'s particle-for-particle walk can pair at most one of
/// the restriction's particles against it before running out of base
/// particles to advance to (`Group_w3c/groupB003v`).
fn flatten(compositor: Compositor, particles: &[Particle]) -> Vec<Particle> {
    let mut flat = Vec::with_capacity(particles.len());
    for particle in particles {
        if particle.min_occurs == 1
            && particle.max_occurs == MaxOccurs::Bounded(1)
            && let Term::Group(group) = &particle.term
            && group.compositor == compositor
        {
            flat.extend(flatten(compositor, &group.particles));
        } else {
            flat.push(particle.clone());
        }
    }
    flat
}

fn compositor_name(compositor: Compositor) -> &'static str {
    match compositor {
        Compositor::Sequence => "sequence",
        Compositor::Choice => "choice",
        Compositor::All => "all",
    }
}

/// `R.min >= B.min` and `R.max <= B.max`.
///
/// XSD 1.1 §3.9.6 (*Occurrence Range OK*) carves out one exception: a
/// particle restricted to occur exactly *zero* times is always a valid
/// restriction of any occurrence range whatsoever, however narrow — that is
/// how a restriction can prohibit a particle its base requires outright
/// rather than merely narrowing how often it may appear
/// (`ModelGroups_w3c/mgH014`: `<xs:element name="e1" minOccurs="0"
/// maxOccurs="0"/>` restricting a required, exactly-once `e1`). It occurs
/// zero times either way, so what its own base counterpart would have
/// otherwise demanded is moot.
fn occurrence_ok(
    schema: &Schema,
    derived: &Particle,
    base: &Particle,
    what: &str,
) -> Result<(), SchemaError> {
    if derived.min_occurs == 0 && derived.max_occurs == MaxOccurs::Bounded(0) {
        return Ok(());
    }
    // XSD 1.1 §3.9.6 states restriction as language containment rather than the
    // 1.0 occurrence-range comparison. A restriction that may be absent is a
    // valid restriction of any base particle that is itself ·emptiable·, even
    // when the base's own `minOccurs` is positive: the base admits the empty
    // sequence through an emptiable branch of its own, so nothing the
    // restriction produces by occurring zero times is outside the base's
    // language (`Additional_w3c/addB118`, whose base `choice` is required once
    // but has an emptiable member). The 1.0 comparison below still holds a 1.0
    // processor to the syntactic rule.
    let base_admits_absence =
        schema.version().is_1_1() && derived.min_occurs == 0 && base.is_emptiable();
    if derived.min_occurs < base.min_occurs && !base_admits_absence {
        return Err(SchemaError::new(
            "rcase-Recurse.1",
            format!(
                "{what}: `minOccurs` {} is below the base's {}",
                derived.min_occurs, base.min_occurs
            ),
        ));
    }
    let ok = match (derived.max_occurs, base.max_occurs) {
        (_, MaxOccurs::Unbounded) => true,
        (MaxOccurs::Unbounded, MaxOccurs::Bounded(_)) => false,
        (MaxOccurs::Bounded(r), MaxOccurs::Bounded(b)) => r <= b,
    };
    if !ok {
        return Err(SchemaError::new(
            "rcase-Recurse.1",
            format!(
                "{what}: `maxOccurs` {} exceeds the base's {}",
                derived.max_occurs, base.max_occurs
            ),
        ));
    }
    Ok(())
}

/// Whether `element` may stand in for `base` wherever a particle referring to
/// `base` admits it — *Substitution Group OK (Transitive)* (§3.3.6.3).
///
/// The chain is walked declaration by declaration rather than read out of the
/// precomputed `{substitution group}` closure, because clause 1 applies at
/// **every** step: a head part-way up the chain that carries
/// `block="substitution"` cannot be stood in for, so nothing below it reaches
/// anything above it. `msData/element/elemZ027b` and `elemZ027c` are the pair
/// that distinguishes the two readings — in the first the blocking declaration
/// is the one doing the substituting, which is no obstacle at all; in the
/// second it sits between the two and breaks the chain.
fn substitutes_for(schema: &Schema, element: ElementId, base: ElementId) -> bool {
    let mut frontier = vec![element];
    let mut seen: Vec<ElementId> = Vec::new();
    while let Some(current) = frontier.pop() {
        if current == base {
            return true;
        }
        if seen.contains(&current) {
            continue;
        }
        seen.push(current);
        for head in &schema.element_declaration(current).substitution_group {
            if schema.element_declaration(*head).block.substitution {
                continue;
            }
            frontier.push(*head);
        }
    }
    false
}

fn name_and_type_ok(
    schema: &Schema,
    derived: &Particle,
    element: ElementId,
    base: &Particle,
    base_element: ElementId,
    what: &str,
) -> Result<(), SchemaError> {
    let r = schema.element_declaration(element);
    let b = schema.element_declaration(base_element);
    if r.name != b.name {
        // §3.9.6 clause 1 is about the *names* only in the simple case. A
        // particle referring to `B` admits every member of B's ·substitution
        // group· as well, so a restriction that narrows the particle to one of
        // those members accepts strictly fewer sequences than the base does,
        // which is what "valid restriction" means
        // (`msData/element/elemZ027b`, `elemZ027e`, `elemZ027f`,
        // `msData/particles/particlesZ008`, where the base particle refers to
        // an *abstract* head that cannot appear in an instance at all).
        //
        // Nothing further is compared once the names differ: the
        // substitution-group rules of §3.3.6 have already established that the
        // member's type derives from the head's and that the derivation is not
        // blocked — the same reasoning `subsume::check_declaration` records for
        // the XSD 1.1 `all`-group case.
        if !substitutes_for(schema, element, base_element) {
            return Err(SchemaError::new(
                "rcase-NameAndTypeOK.1",
                format!("{what}: `{}` does not restrict `{}`", r.name, b.name),
            ));
        }
        // Clause 5.2.1 all the same: the member's type must be derived from the
        // head's "given `{extension, list, union}`" — that is, by restriction
        // only. A member whose type *extends* the head's accepts content the
        // base particle would have refused, so narrowing the particle to it is
        // not a restriction at all (`msData/particles/particlesV020`, where the
        // member's type adds a required element to the head's).
        if b.type_id != schema.any_type() && r.type_id != b.type_id {
            let by_restriction = schema
                .derivation_path(r.type_id, b.type_id)
                .is_some_and(|used| !used.extension);
            if !by_restriction {
                return Err(SchemaError::new(
                    "rcase-NameAndTypeOK.7",
                    format!(
                        "{what}: the type of `{}` is not derived by restriction from its \
                         substitution group head's",
                        r.name
                    ),
                ));
            }
        }
        return occurrence_ok(schema, derived, base, what);
    }
    occurrence_ok(schema, derived, base, what)?;
    if r.nillable && !b.nillable {
        return Err(SchemaError::new(
            "rcase-NameAndTypeOK.2",
            format!("{what}: `{}` is nillable but its base is not", r.name),
        ));
    }
    if let Some(constraint) = &b.value_constraint
        && constraint.fixed
    {
        let matches = r
            .value_constraint
            .as_ref()
            .is_some_and(|own| own.fixed && own.value == constraint.value);
        if !matches {
            return Err(SchemaError::new(
                "rcase-NameAndTypeOK.4",
                format!("{what}: `{}` is fixed in the base type", r.name),
            ));
        }
    }
    // §3.9.6 *rcase-NameAndTypeOK.6*: `R`'s `{disallowed substitutions}` must
    // be a superset of `B`'s, so a restriction may not re-open a substitution
    // the base blocked (`msData/particles/particlesL014`).
    if (b.block.extension && !r.block.extension)
        || (b.block.restriction && !r.block.restriction)
        || (b.block.substitution && !r.block.substitution)
    {
        return Err(SchemaError::new(
            "rcase-NameAndTypeOK.6",
            format!(
                "{what}: `{}` blocks less than the same element does in the base type",
                r.name
            ),
        ));
    }
    // *rcase-NameAndTypeOK.5*: every identity constraint of `B` must also be
    // one of `R`.
    if b.identity_constraints
        .iter()
        .any(|constraint| !r.identity_constraints.contains(constraint))
    {
        return Err(SchemaError::new(
            "rcase-NameAndTypeOK.5",
            format!(
                "{what}: `{}` drops an identity constraint the base type declares",
                r.name
            ),
        ));
    }
    // *rcase-NameAndTypeOK.7*: `R`'s type must be validly derived from `B`'s
    // **given `{extension, list, union}`** — that is, by restriction. A type
    // that reaches the base by extending it accepts content the base particle
    // would have refused, so narrowing the particle to it is not a restriction
    // at all (`msData/particles/particlesIj008`, where the restriction gives
    // the element a type that *extends* the one the base gives it).
    //
    // `list` and `union` are left out of the blocked set here because
    // §3.14.6 clause 2.2.3 makes a list or union type validly derived from
    // the ·simple ur-type definition· whatever the subset says, and that is
    // the only way either method can appear on a path this check sees.
    if r.type_id != b.type_id {
        let by_restriction =
            derives_without_extension(schema, r.type_id, b.type_id, schema.type_count());
        if !by_restriction {
            return Err(SchemaError::new(
                "rcase-NameAndTypeOK.7",
                format!(
                    "{what}: the type of `{}` is not derived by restriction from its type in \
                     the base",
                    r.name
                ),
            ));
        }
    }
    type_table_restricts(schema, r, b, what)
}

/// XSD 1.1 §3.9.6 (*Conditional Type Substitutable in Restriction*): when the
/// base declaration carries a {type table}, the restricting declaration's has
/// to select, for every element information item, a type validly derived by
/// restriction from the one the base's would have selected.
///
/// The tests are XPath expressions, so which of them an item satisfies cannot
/// be decided here. What is decidable — and what this checks — is the case
/// where the two tables ask the same questions in the same order: the item
/// that answers the *n*th question in the restriction is then the one that
/// answers the *n*th in the base, so each answer has to be a restriction of
/// the base's answer, and a table that asks different questions altogether is
/// refused rather than guessed at. `saxonData/CTA/cta0043` is the group that
/// pins this down: its two tables differ in one alternative, where the
/// restriction offers `xs:dateTime` content and the base `xs:dateTimeStamp` —
/// a widening, not a narrowing.
fn type_table_restricts(
    schema: &Schema,
    r: &crate::components::ElementDeclaration,
    b: &crate::components::ElementDeclaration,
    what: &str,
) -> Result<(), SchemaError> {
    if !schema.version().is_1_1() || b.alternatives.is_empty() {
        return Ok(());
    }
    let mismatch = |detail: &str| {
        Err(SchemaError::new(
            "rcase-NameAndTypeOK",
            format!(
                "{what}: the type table of `{}` does not restrict the one the base type gives \
                 it — {detail}",
                r.name
            ),
        ))
    };
    if r.alternatives.len() != b.alternatives.len() {
        return mismatch("they offer different numbers of alternatives");
    }
    for (own, theirs) in r.alternatives.iter().zip(&b.alternatives) {
        let same_test = match (&own.test, &theirs.test) {
            (Some(own), Some(theirs)) => own.test == theirs.test,
            (None, None) => true,
            _ => false,
        };
        if !same_test {
            return mismatch("they test different conditions");
        }
        if own.type_id != theirs.type_id
            && !derives_without_extension(schema, own.type_id, theirs.type_id, schema.type_count())
        {
            return mismatch("an alternative widens the type the base selects");
        }
    }
    Ok(())
}

/// Whether `candidate` is validly ·derived· from `base` given the blocking
/// subset `{extension}` — §3.4.6 *Type Derivation OK (Complex)* clause 1 and
/// §3.16.6.3 *Type Derivation OK (Simple)*.
///
/// This is [`Schema::is_derived_from`] with the one extra condition that no
/// step of the chain is an extension, and it keeps that function's clause
/// 2.2.4: a type derived from one *member* of an unconstrained union is
/// derived from the union itself (`msData/additional/addB150`,
/// `saxonData/Simple/simple010`, where a restriction narrows an element whose
/// type in the base is a union to one of that union's members).
fn derives_without_extension(
    schema: &Schema,
    candidate: TypeId,
    base: TypeId,
    budget: usize,
) -> bool {
    if candidate == base {
        return true;
    }
    if budget > 0
        && let Some(members) = schema.union_member_types(base)
        && schema
            .simple_type(base)
            .is_some_and(|simple| simple.facets.is_unconstrained())
        && members
            .iter()
            .any(|member| derives_without_extension(schema, candidate, *member, budget - 1))
    {
        return true;
    }
    let mut current = candidate;
    let mut guard = 0;
    loop {
        if current == base {
            return true;
        }
        let definition = schema.type_definition(current);
        if definition.derivation == Derivation::Extension {
            return false;
        }
        let Some(next) = definition.base else {
            return false;
        };
        if next == current {
            return false;
        }
        current = next;
        guard += 1;
        if guard > schema.type_count() {
            return false;
        }
    }
}

fn ns_compat(
    schema: &Schema,
    derived: &Particle,
    element: ElementId,
    base: &Particle,
    wildcard: &Wildcard,
    what: &str,
) -> Result<(), SchemaError> {
    let name = &schema.element_declaration(element).name;
    if !wildcard.namespace.allows(name.namespace()) {
        return Err(SchemaError::new(
            "rcase-NSCompat.1",
            format!("{what}: the base wildcard does not admit `{name}`"),
        ));
    }
    occurrence_ok(schema, derived, base, what)
}

fn ns_subset(
    schema: &Schema,
    derived: &Particle,
    r: &Wildcard,
    base: &Particle,
    b: &Wildcard,
    what: &str,
) -> Result<(), SchemaError> {
    occurrence_ok(schema, derived, base, what)?;
    if !b.namespace.is_superset_of(&r.namespace) {
        return Err(SchemaError::new(
            "rcase-NSSubset.2",
            format!("{what}: the wildcard is not a subset of the base's"),
        ));
    }
    if process_rank(r.process_contents) < process_rank(b.process_contents) {
        return Err(SchemaError::new(
            "rcase-NSSubset.3",
            format!("{what}: the wildcard weakens the base's `processContents`"),
        ));
    }
    Ok(())
}

fn process_rank(process: ProcessContents) -> u8 {
    match process {
        ProcessContents::Skip => 0,
        ProcessContents::Lax => 1,
        ProcessContents::Strict => 2,
    }
}

fn ns_recurse_check_cardinality(
    schema: &Schema,
    derived: &Particle,
    group: &ModelGroup,
    base: &Particle,
    wildcard: &Wildcard,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    let unbounded_base = Particle {
        min_occurs: 0,
        max_occurs: MaxOccurs::Unbounded,
        term: Term::Wildcard(wildcard.clone()),
    };
    for particle in &group.particles {
        particle_valid_restriction(schema, particle, &unbounded_base, what, depth + 1)?;
    }
    // The effective total range of the group must fit inside the base's.
    let (min, max) = effective_total_range(derived, group, depth);
    let bounded = match (max, base.max_occurs) {
        (_, MaxOccurs::Unbounded) => true,
        (MaxOccurs::Unbounded, MaxOccurs::Bounded(_)) => false,
        (MaxOccurs::Bounded(r), MaxOccurs::Bounded(b)) => r <= b,
    };
    if min < base.min_occurs || !bounded {
        return Err(SchemaError::new(
            "rcase-NSRecurseCheckCardinality.2",
            format!("{what}: the group's occurrence range exceeds the base wildcard's"),
        ));
    }
    Ok(())
}

/// The *effective total range* of a group particle (§3.8.6).
///
/// §3.8.6 states the range over "the {min occurs} of every wildcard or element
/// declaration particle in the group's {particles} **and the minimum part of
/// the effective total range of each of the group particles**", so a member
/// that is itself a group contributes the range of everything inside it rather
/// than its own occurrence range alone. A group reference is exactly such a
/// member — the component model gives it no wrapping beyond the referenced
/// model group — so without the recursion a restriction wrapping two group
/// references looks as though it could match twice when it can match four
/// times (`msData/particles/particlesQ012`).
fn effective_total_range(
    particle: &Particle,
    group: &ModelGroup,
    depth: usize,
) -> (u32, MaxOccurs) {
    let member = |member: &Particle| member_total_range(member, depth + 1);
    let (inner_min, inner_max) = match group.compositor {
        Compositor::Choice => {
            let min = group
                .particles
                .iter()
                .map(|part| member(part).0)
                .min()
                .unwrap_or(0);
            let max = group
                .particles
                .iter()
                .fold(MaxOccurs::Bounded(0), |acc, part| {
                    match (acc, member(part).1) {
                        (MaxOccurs::Unbounded, _) | (_, MaxOccurs::Unbounded) => {
                            MaxOccurs::Unbounded
                        }
                        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => {
                            MaxOccurs::Bounded(a.max(b))
                        }
                    }
                });
            (min, max)
        }
        Compositor::Sequence | Compositor::All => {
            let min = group
                .particles
                .iter()
                .fold(0_u32, |acc, part| acc.saturating_add(member(part).0));
            let max = group
                .particles
                .iter()
                .fold(MaxOccurs::Bounded(0), |acc, part| {
                    match (acc, member(part).1) {
                        (MaxOccurs::Unbounded, _) | (_, MaxOccurs::Unbounded) => {
                            MaxOccurs::Unbounded
                        }
                        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => {
                            MaxOccurs::Bounded(a.saturating_add(b))
                        }
                    }
                });
            (min, max)
        }
    };
    let min = inner_min.saturating_mul(particle.min_occurs);
    let max = match (inner_max, particle.max_occurs) {
        (MaxOccurs::Unbounded, _) | (_, MaxOccurs::Unbounded) => MaxOccurs::Unbounded,
        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => MaxOccurs::Bounded(a.saturating_mul(b)),
    };
    (min, max)
}

/// One member's contribution to an [`effective_total_range`]: its own
/// occurrence range, unless it is a group particle, whose whole effective
/// total range counts instead.
///
/// The recursion is bounded by the same [`DEPTH_LIMIT`] the restriction walk
/// uses; past it the member counts as itself, which can only make the computed
/// range smaller and so can only *accept*.
fn member_total_range(particle: &Particle, depth: usize) -> (u32, MaxOccurs) {
    match &particle.term {
        Term::Group(group) if depth <= DEPTH_LIMIT => effective_total_range(particle, group, depth),
        _ => (particle.min_occurs, particle.max_occurs),
    }
}

/// `Recurse` (strict) and `RecurseLax` (lenient about skipped base particles).
///
/// The mapping has to preserve order but need not be greedy — `(a)` is a valid
/// restriction of `(a?, a)` only by skipping the *first* base particle — so the
/// search back-tracks, bounded by a step budget.
/// *RecurseLax* without the order-preservation clause, which is how a `choice`
/// restricting a `choice` comes out under XSD 1.1.
///
/// XSD 1.0 asks for a complete **order-preserving** functional mapping from the
/// restriction's branches to the base's. A `choice` accepts the union of its
/// branches' languages and nothing about that union depends on the order the
/// branches are written in, so §3.9.6.2's statement about languages does not
/// impose the clause: if every branch of the restriction is a ·valid
/// restriction· of *some* branch of the base — whichever, in whatever order —
/// then every sequence the restriction accepts is one the base accepts too.
/// `Particles_w3c/particlesT002` and `particlesT009` are the pair that pins
/// this down: their base and restriction offer the same two branches written
/// the other way round, and the suite marks them invalid for 1.0 and valid for
/// 1.1.
///
/// The mapping need not be injective — two branches of the restriction may map
/// to one of the base's — which is why this is a plain "each of these is
/// accepted by one of those" rather than a matching problem.
fn recurse_lax_unordered(
    schema: &Schema,
    derived: &[Particle],
    base: &[Particle],
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    for particle in derived {
        if !base.iter().any(|candidate| {
            particle_valid_restriction(schema, particle, candidate, what, depth + 1).is_ok()
        }) {
            return Err(SchemaError::new(
                "rcase-RecurseLax.2",
                format!("{what}: a branch of the choice restricts no branch of the base choice"),
            ));
        }
    }
    Ok(())
}

fn recurse(
    schema: &Schema,
    derived: &[Particle],
    base: &[Particle],
    what: &str,
    depth: usize,
    skipped_must_be_emptiable: bool,
) -> Result<(), SchemaError> {
    let mut budget = 4_000_usize;
    if recurse_from(
        schema,
        derived,
        base,
        0,
        0,
        what,
        depth,
        skipped_must_be_emptiable,
        &mut budget,
    ) {
        Ok(())
    } else if budget == 0 {
        // The search ran out of steps: accept rather than reject a schema that
        // may well be legal.
        Ok(())
    } else if schema.version().is_1_1()
        && recurse_by_branches(
            schema,
            derived,
            base,
            what,
            depth,
            skipped_must_be_emptiable,
        )
    {
        Ok(())
    } else {
        Err(SchemaError::new(
            "rcase-Recurse.2",
            format!("{what}: the content model does not map onto the base type's"),
        ))
    }
}

/// The number of one-iteration alternatives [`recurse_by_branches`] will try
/// before giving up on the decomposition.
const BRANCH_BUDGET: usize = 64;

/// XSD 1.1's language reading of §3.9.6, for a restriction whose members
/// include a `choice`.
///
/// One pass over a sequence that holds a `choice` picks exactly one of its
/// branches, so the language of that pass is the *union* of the languages of
/// the sequences that write one branch in the `choice`'s place. A union is
/// contained in the base's language exactly when each part is, so the members
/// are expanded into those alternatives and each is put through the ordinary
/// mapping. The occurrence range of the enclosing particles is checked by the
/// caller before this is reached, which is what makes the decomposition sound
/// for a group that repeats: if one pass of the restriction is accepted by one
/// pass of the base and the restriction's range is within the base's, then so
/// is every repetition.
///
/// XSD 1.0 asks instead for a mapping from each member of the restriction to a
/// member of the base, which no `choice` spanning several base members can
/// satisfy — hence the version split the suite records for
/// `msData/particles/particlesHb008`.
///
/// Only a `choice` that occurs exactly once is expanded: for any other range
/// the alternatives interleave and their union is smaller than the language it
/// would stand for. The expansion is abandoned past [`BRANCH_BUDGET`]
/// alternatives, leaving the caller's original diagnostic in place.
fn recurse_by_branches(
    schema: &Schema,
    derived: &[Particle],
    base: &[Particle],
    what: &str,
    depth: usize,
    skipped_must_be_emptiable: bool,
) -> bool {
    let mut alternatives: Vec<Vec<Particle>> = vec![Vec::new()];
    let mut expanded = false;
    for member in derived {
        let branches: Vec<&Particle> = match &member.term {
            Term::Group(group)
                if group.compositor == Compositor::Choice
                    && member.min_occurs == 1
                    && member.max_occurs == MaxOccurs::Bounded(1)
                    && !group.particles.is_empty() =>
            {
                expanded = true;
                group.particles.iter().collect()
            }
            _ => vec![member],
        };
        if alternatives.len().saturating_mul(branches.len()) > BRANCH_BUDGET {
            return false;
        }
        alternatives = alternatives
            .into_iter()
            .flat_map(|prefix| {
                branches.iter().map(move |branch| {
                    let mut next = prefix.clone();
                    next.push((*branch).clone());
                    next
                })
            })
            .collect();
    }
    if !expanded {
        return false;
    }
    alternatives.iter().all(|alternative| {
        let mut budget = 4_000_usize;
        recurse_from(
            schema,
            alternative,
            base,
            0,
            0,
            what,
            depth,
            skipped_must_be_emptiable,
            &mut budget,
        ) || budget == 0
    })
}

#[allow(clippy::too_many_arguments)]
fn recurse_from(
    schema: &Schema,
    derived: &[Particle],
    base: &[Particle],
    di: usize,
    bi: usize,
    what: &str,
    depth: usize,
    strict: bool,
    budget: &mut usize,
) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    if di == derived.len() {
        return !strict
            || base[bi.min(base.len())..]
                .iter()
                .all(Particle::is_emptiable);
    }
    let mut index = bi;
    while index < base.len() {
        if particle_valid_restriction(schema, &derived[di], &base[index], what, depth + 1).is_ok() {
            if recurse_from(
                schema,
                derived,
                base,
                di + 1,
                index + 1,
                what,
                depth,
                strict,
                budget,
            ) {
                return true;
            }
            // XSD 1.1 §3.9.6: an unbounded base particle can be the image of
            // several consecutive restriction particles. Its language is closed
            // under concatenation, so once one restriction particle is a valid
            // restriction of it, the next may map onto the same base particle
            // rather than being forced past it — which is how a base that
            // factors a repeated choice out into a group (spliced back in here by
            // `flatten`) still admits a restriction that lists the choice's
            // members inline (`Particles_w3c/particlesZ028`).
            if schema.version().is_1_1()
                && base[index].max_occurs == MaxOccurs::Unbounded
                && recurse_from(
                    schema,
                    derived,
                    base,
                    di + 1,
                    index,
                    what,
                    depth,
                    strict,
                    budget,
                )
            {
                return true;
            }
        }
        if strict && !base[index].is_emptiable() {
            return false;
        }
        index += 1;
    }
    false
}

/// `RecurseUnordered`: a sequence restricting an `all` group.
fn recurse_unordered(
    schema: &Schema,
    derived: &[Particle],
    base: &[Particle],
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    let mut used = vec![false; base.len()];
    for particle in derived {
        let mut matched = false;
        for (index, candidate) in base.iter().enumerate() {
            if used[index] {
                continue;
            }
            if particle_valid_restriction(schema, particle, candidate, what, depth + 1).is_ok() {
                used[index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return Err(SchemaError::new(
                "rcase-RecurseUnordered.2",
                format!("{what}: a particle has no counterpart in the base `all` group"),
            ));
        }
    }
    for (index, candidate) in base.iter().enumerate() {
        if !used[index] && !candidate.is_emptiable() {
            return Err(SchemaError::new(
                "rcase-RecurseUnordered.2",
                format!("{what}: a required particle of the base `all` group is dropped"),
            ));
        }
    }
    Ok(())
}

/// `MapAndSum`: a sequence restricting a choice.
fn map_and_sum(
    schema: &Schema,
    derived: &Particle,
    derived_group: &ModelGroup,
    base: &Particle,
    base_group: &ModelGroup,
    what: &str,
    depth: usize,
) -> Result<(), SchemaError> {
    // XSD 1.1 §3.9.6 restates *Particle Valid (Restriction)* as language
    // containment. When the base `choice` may occur exactly once — so a single
    // one of its branches can stand for a whole occurrence of it — a restriction
    // that is itself a ·valid restriction· of one branch accepts only sequences
    // that branch, and therefore the choice, already accepts. This is what makes
    // a `choice` restricting a `choice` keeping a single branch (which `simplify`
    // rewrites as that branch, so the pair arrives here as sequence-restricts-
    // choice) a valid restriction under 1.1, even though the 1.0 `MapAndSum` sum
    // below rejects it (`Particles_w3c/particlesZ023`, `particlesZ024`,
    // `Group_w3c/groupH021v`). The guard keeps this sound: were the base choice
    // able to force two or more picks, one branch could no longer stand for the
    // whole.
    if schema.version().is_1_1()
        && base.min_occurs <= 1
        && !matches!(base.max_occurs, MaxOccurs::Bounded(0))
    {
        let branches = flatten(base_group.compositor, &base_group.particles);
        if branches.iter().any(|branch| {
            particle_valid_restriction(schema, derived, branch, what, depth + 1).is_ok()
        }) {
            return Ok(());
        }
    }
    let derived_particles = flatten(derived_group.compositor, &derived_group.particles);
    let base_particles = flatten(base_group.compositor, &base_group.particles);
    for particle in &derived_particles {
        let matched = base_particles.iter().any(|candidate| {
            particle_valid_restriction(schema, particle, candidate, what, depth + 1).is_ok()
        });
        if !matched {
            return Err(SchemaError::new(
                "rcase-MapAndSum.1",
                format!("{what}: a particle matches no branch of the base choice"),
            ));
        }
    }
    let count = u32::try_from(derived_particles.len()).unwrap_or(u32::MAX);
    let min = derived.min_occurs.saturating_mul(count);
    let max = match derived.max_occurs {
        MaxOccurs::Unbounded => MaxOccurs::Unbounded,
        MaxOccurs::Bounded(value) => MaxOccurs::Bounded(value.saturating_mul(count)),
    };
    if min < base.min_occurs {
        return Err(SchemaError::new(
            "rcase-MapAndSum.2",
            format!("{what}: the sequence cannot occur as often as the base choice requires"),
        ));
    }
    let bounded = match (max, base.max_occurs) {
        (_, MaxOccurs::Unbounded) => true,
        (MaxOccurs::Unbounded, MaxOccurs::Bounded(_)) => false,
        (MaxOccurs::Bounded(r), MaxOccurs::Bounded(b)) => r <= b,
    };
    if !bounded {
        return Err(SchemaError::new(
            "rcase-MapAndSum.2",
            format!("{what}: the sequence can occur more often than the base choice allows"),
        ));
    }
    Ok(())
}

/// *Particle Valid (Restriction)* between two free-standing particles, used by
/// the `xs:redefine` rules of §4.2.2.
pub(crate) fn particle_restricts(
    schema: &Schema,
    derived: &Particle,
    base: &Particle,
    what: &str,
) -> Result<(), SchemaError> {
    particle_valid_restriction(schema, derived, base, what, 0)
}

/// The attribute half of the same, for `xs:attributeGroup` redefinitions:
/// every use the redefinition keeps must narrow the original's, and no
/// required use of the original may be dropped or added to.
pub(crate) fn attribute_uses_restrict(
    schema: &Schema,
    derived: &crate::components::AttributeGroupDefinition,
    base: &crate::components::AttributeGroupDefinition,
    what: &str,
) -> Result<(), SchemaError> {
    for use_ in &derived.attribute_uses {
        // §3.6.2: an `<attribute use="prohibited">` inside an attribute group
        // definition contributes no attribute use at all — on either side. A
        // redefinition that prohibits what the original prohibited adds
        // nothing and needs no counterpart (`msData/schema/schT3`), while one
        // that *offers* an attribute the original prohibits is adding one
        // rather than restricting it (`msData/attributeGroup/attgZ003`).
        if use_.kind == AttributeUseKind::Prohibited {
            continue;
        }
        let declaration = schema.attribute_declaration(use_.attribute);
        let Some(base_use) = base.attribute_uses.iter().find(|candidate| {
            candidate.kind != AttributeUseKind::Prohibited
                && schema.attribute_declaration(candidate.attribute).name == declaration.name
        }) else {
            let admitted = base
                .attribute_wildcard
                .as_ref()
                .is_some_and(|wildcard| wildcard.namespace.allows(declaration.name.namespace()));
            if admitted {
                continue;
            }
            return Err(SchemaError::new(
                "src-redefine.7.2.2",
                format!("{what}: it adds the attribute `{}`", declaration.name),
            ));
        };
        let base_type = schema.attribute_declaration(base_use.attribute).type_id;
        if base_type != declaration.type_id
            && !schema.is_derived_from(declaration.type_id, base_type)
        {
            return Err(SchemaError::new(
                "src-redefine.7.2.2",
                format!(
                    "{what}: the type of `{}` is not derived from its type in the original",
                    declaration.name
                ),
            ));
        }
        // *src-redefine.7.2.2* points at clause 2 of *Derivation Valid
        // (Restriction, Complex)*, whose clause 2.1.3 keeps a fixed value
        // fixed: a redefinition may not turn the original's `fixed` into a
        // `default`, or into no value constraint at all, because instances
        // written against the original are entitled to rely on it
        // (`msData/schema/schM10`).
        let base_constraint = base_use.value_constraint.as_ref().or(schema
            .attribute_declaration(base_use.attribute)
            .value_constraint
            .as_ref());
        if let Some(base_constraint) = base_constraint
            && base_constraint.fixed
        {
            let own = use_
                .value_constraint
                .as_ref()
                .or(declaration.value_constraint.as_ref());
            if !own.is_some_and(|own| own.fixed && own.value == base_constraint.value) {
                return Err(SchemaError::new(
                    "src-redefine.7.2.2",
                    format!("{what}: `{}` is fixed in the original", declaration.name),
                ));
            }
        }
    }
    for base_use in &base.attribute_uses {
        if base_use.kind != AttributeUseKind::Required {
            continue;
        }
        let name = &schema.attribute_declaration(base_use.attribute).name;
        if !derived
            .attribute_uses
            .iter()
            .any(|candidate| &schema.attribute_declaration(candidate.attribute).name == name)
        {
            return Err(SchemaError::new(
                "src-redefine.7.2.2",
                format!("{what}: it drops the required attribute `{name}`"),
            ));
        }
    }
    Ok(())
}
