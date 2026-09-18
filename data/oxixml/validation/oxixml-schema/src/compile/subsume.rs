//! The XSD **1.1** restriction rules for content models built on `xs:all`.
//!
//! XSD 1.0 decided *Particle Valid (Restriction)* with a table of syntactic
//! cases (`rcase-Recurse`, `rcase-RecurseUnordered`, `rcase-MapAndSum`, …), one
//! cell per pair of compositors, and left most cells that involve an `xs:all`
//! group either forbidden or handled as if the group were ordered. XSD 1.1
//! §3.9.6 replaced that table with a semantic statement — every sequence
//! ·valid· with respect to `R` must be ·valid· with respect to `B`, and each
//! item must be governed by a declaration that is compatible with the one `B`
//! would have used — and the 1.1 test suite exercises exactly the cases the
//! 1.0 table got wrong: occurrence ranges changed inside an `all`, a `sequence`
//! or `choice` restricting an `all`, and members mapped onto a base member's
//! substitution group.
//!
//! This module decides that statement **exactly** for the case where the base
//! particle is an `all` group, which is where all of those cases live.
//!
//! # Why the decision is exact
//!
//! *Unique Particle Attribution* (§3.8.6.4) makes the attribution of an item to
//! a member of an `all` group a function of the item's **name** alone — XSD 1.1
//! makes wildcards *weak*, so a name an element declaration member declares
//! goes to that member and never to a wildcard member, and the remaining
//! wildcard members are pairwise disjoint. So the language of the group is
//! precisely
//!
//! > every sequence in which, for each member `j`, the number of items
//! > attributed to `j` lies in `[min_j, max_j]`.
//!
//! Membership is therefore decided coordinate by coordinate, and a set of
//! sequences is contained in the language exactly when, for each `j`, *every*
//! count the restriction can produce for `j` lies in `[min_j, max_j]`. Since
//! the target is an interval, only the least and the greatest such count
//! matter, and [`counts`] computes both by structural recursion over the
//! restricting particle.
//!
//! The one thing that recursion has to be careful about is a **wildcard leaf**,
//! whose items do not all go to one member: it produces names, and each name is
//! attributed on its own. [`leaf_distribution`] therefore answers, per member,
//! whether *every* name the leaf can produce is attributed there
//! ([`Reach::Always`]), only *some* are ([`Reach::Sometimes`]), or none is
//! ([`Reach::Never`]) — which is exactly what turns the leaf's own occurrence
//! range into a per-member one. Two consequences are easy to get wrong and are
//! what the fixtures below are about:
//!
//! * a wildcard leaf reaching a member only *sometimes* can contribute **none**
//!   of its items there, however large its own `minOccurs` is — that is why
//!   `saxonData/All/all244` (a restricting wildcard straddling two base
//!   members, so neither is guaranteed the five items one of them requires) and
//!   `saxonData/Wild/wild069` (a restricting wildcard that the base would
//!   sometimes attribute to an element declaration, leaving its own wildcard
//!   member with nothing) are both invalid restrictions;
//! * a wildcard leaf does not compete for the names its **own** siblings
//!   declare, because within the restriction those names are attributed to the
//!   sibling element particle — without that, `saxonData/Wild/wild030` and
//!   `wild047` would be reported as widening a member that they leave alone.

use std::collections::BTreeSet;

use oxixml_qname::ExpandedName;

use crate::components::{
    Compositor, MaxOccurs, ModelGroup, NamespaceConstraint, Particle, Term, Wildcard,
};
use crate::error::SchemaError;
use crate::schema::Schema;

/// One member of the (flattened) base `all` group.
struct Member<'a> {
    term: &'a Term,
    min_occurs: u32,
    max_occurs: MaxOccurs,
}

/// The least and greatest number of items a particle can contribute to one
/// member of the base group.
type Range = (u32, MaxOccurs);

const ZERO: Range = (0, MaxOccurs::Bounded(0));

/// Whether XSD 1.1's subsumption rules should decide this pair.
pub(crate) fn applies(schema: &Schema, base: &Particle) -> bool {
    schema.version().is_1_1()
        && matches!(&base.term, Term::Group(group) if group.compositor == Compositor::All)
}

/// Decides *Particle Valid (Restriction)* for a base particle whose term is an
/// `all` group, under XSD 1.1.
pub(crate) fn all_restriction(
    schema: &Schema,
    derived: &Particle,
    base: &Particle,
    what: &str,
) -> Result<(), SchemaError> {
    let Term::Group(group) = &base.term else {
        return Ok(());
    };
    let mut members = Vec::new();
    if flatten(group, &mut members).is_err() {
        // A base group this module cannot read is left to the 1.0 cases.
        return Ok(());
    }
    // The occurrence range of the restricting particle *itself* is compared
    // with the base group's, exactly as the 1.0 cases do; the per-member counts
    // below then describe one occurrence of it. Folding the two together would
    // make an optional `all` restricting an optional `all` look as if it
    // dropped every required member (`msData/modelGroups/mgO029`).
    let derived = &match &derived.term {
        Term::Group(_) => {
            if derived.min_occurs < base.min_occurs || !fits(derived.max_occurs, base.max_occurs) {
                return Err(SchemaError::new(
                    "rcase-Recurse.1",
                    format!("{what}: the occurrence range exceeds the base `all` group's"),
                ));
            }
            Particle {
                min_occurs: 1,
                max_occurs: MaxOccurs::Bounded(1),
                term: derived.term.clone(),
            }
        }
        _ => derived.clone(),
    };
    let vector = counts(schema, derived, &members, &[], what)?;
    for (index, member) in members.iter().enumerate() {
        let (min, max) = vector.get(index).copied().unwrap_or(ZERO);
        // Each member's own range must be respected, whether or not the base
        // group as a whole is optional: an emptiable base group does not let
        // the restriction drop a required member and keep the others
        // (`msData/particles/particlesK006`).
        if min < member.min_occurs {
            return Err(SchemaError::new(
                "rcase-Recurse.2",
                format!(
                    "{what}: the restriction can supply as few as {min} of {}, which the base \
                     `all` group requires at least {} of",
                    describe(member.term),
                    member.min_occurs
                ),
            ));
        }
        if !fits(max, member.max_occurs) {
            return Err(SchemaError::new(
                "rcase-Recurse.2",
                format!(
                    "{what}: the restriction can supply as many as {max} of {}, which the base \
                     `all` group allows at most {} of",
                    describe(member.term),
                    member.max_occurs
                ),
            ));
        }
    }
    Ok(())
}

/// Flattens a base `all` group, following the XSD 1.1 rule that a member may
/// itself be an `all` group (a reference to a named group whose compositor is
/// `all`).
fn flatten<'a>(group: &'a ModelGroup, members: &mut Vec<Member<'a>>) -> Result<(), ()> {
    for particle in &group.particles {
        match &particle.term {
            Term::Element(_) | Term::Wildcard(_) => members.push(Member {
                term: &particle.term,
                min_occurs: particle.min_occurs,
                max_occurs: particle.max_occurs,
            }),
            Term::Group(nested)
                if nested.compositor == Compositor::All
                    && particle.min_occurs == 1
                    && particle.max_occurs == MaxOccurs::Bounded(1) =>
            {
                flatten(nested, members)?;
            }
            Term::Group(_) => return Err(()),
        }
    }
    Ok(())
}

/// The least and greatest number of items `particle` can contribute to each
/// member of the base group.
///
/// `siblings` holds the names the model group `particle` is written in already
/// attributes to an element particle of its own, which a wildcard leaf
/// therefore never sees.
///
/// Errors when a leaf of `particle` matches no member at all: the restriction
/// then accepts an element the base does not.
fn counts(
    schema: &Schema,
    particle: &Particle,
    members: &[Member<'_>],
    siblings: &[ExpandedName],
    what: &str,
) -> Result<Vec<Range>, SchemaError> {
    let mut vector = vec![ZERO; members.len()];
    match &particle.term {
        Term::Element(_) | Term::Wildcard(_) => {
            let reach = leaf_distribution(schema, &particle.term, members, siblings, what)?;
            for (slot, reach) in vector.iter_mut().zip(reach) {
                *slot = match reach {
                    Reach::Never => ZERO,
                    // A member the leaf reaches only sometimes may receive
                    // every item it produces, or none of them: the names that
                    // would have gone elsewhere are the leaf's to choose.
                    Reach::Sometimes => (0, particle.max_occurs),
                    Reach::Always => (particle.min_occurs, particle.max_occurs),
                };
            }
            return Ok(vector);
        }
        Term::Group(group) => {
            let declared = declared_names(schema, group);
            let children: Vec<Vec<Range>> = group
                .particles
                .iter()
                .map(|child| counts(schema, child, members, &declared, what))
                .collect::<Result<_, _>>()?;
            match group.compositor {
                Compositor::Sequence | Compositor::All => {
                    for child in &children {
                        for (index, slot) in vector.iter_mut().enumerate() {
                            let (min, max) = child.get(index).copied().unwrap_or(ZERO);
                            slot.0 = slot.0.saturating_add(min);
                            slot.1 = add(slot.1, max);
                        }
                    }
                }
                Compositor::Choice => {
                    if children.is_empty() {
                        return Ok(vector);
                    }
                    for (index, slot) in vector.iter_mut().enumerate() {
                        let mut least = u32::MAX;
                        let mut greatest = MaxOccurs::Bounded(0);
                        for child in &children {
                            let (min, max) = child.get(index).copied().unwrap_or(ZERO);
                            least = least.min(min);
                            greatest = larger(greatest, max);
                        }
                        *slot = (least, greatest);
                    }
                }
            }
        }
    }
    for slot in &mut vector {
        slot.0 = slot.0.saturating_mul(particle.min_occurs);
        slot.1 = times(slot.1, particle.max_occurs);
    }
    Ok(vector)
}

/// How many of a leaf's items one member of the base group can be given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// None: no name the leaf produces is attributed to this member.
    Never,
    /// Some: the leaf can send every item here, or none at all.
    Sometimes,
    /// All: every name the leaf produces is attributed to this member.
    Always,
}

/// The names a model group of the *restriction* attributes to an element
/// particle of its own, and which a wildcard in the same group therefore never
/// sees.
///
/// An element particle matches its whole substitution group, so the members of
/// one are named here beside the declaration's own name.
fn declared_names(schema: &Schema, group: &ModelGroup) -> Vec<ExpandedName> {
    let mut names = Vec::new();
    for particle in &group.particles {
        if let Term::Element(element) = &particle.term {
            let declaration = schema.element_declaration(*element);
            names.push(declaration.name.clone());
            for candidate in &declaration.substitutable_by {
                names.push(schema.element_declaration(*candidate).name.clone());
            }
        }
    }
    names
}

/// Whether `wildcard`, written beside `siblings`, can produce `name`.
fn admits(
    schema: &Schema,
    wildcard: &Wildcard,
    name: &ExpandedName,
    siblings: &[ExpandedName],
) -> bool {
    if siblings.contains(name) {
        return false;
    }
    let defined = wildcard.disallowed.defined
        && schema
            .find_element(name.namespace(), name.local_name())
            .is_some();
    wildcard.matches(name.namespace(), name.local_name(), defined)
}

/// How a leaf of the restriction distributes over the members of the base
/// group.
///
/// XSD 1.1 §3.8.6.4 makes wildcards *weak*, so an element declaration member is
/// preferred over a wildcard member — the same order the validator attributes
/// items in.
///
/// # Errors
///
/// Reports the leaf as widening the base when the base group has no member at
/// all for some name the leaf can produce.
fn leaf_distribution(
    schema: &Schema,
    term: &Term,
    members: &[Member<'_>],
    siblings: &[ExpandedName],
    what: &str,
) -> Result<Vec<Reach>, SchemaError> {
    let mut reach = vec![Reach::Never; members.len()];
    match term {
        Term::Element(element) => {
            let name = &schema.element_declaration(*element).name;
            let matches_declaration = |member: &Member<'_>| match member.term {
                Term::Element(base) => {
                    let declaration = schema.element_declaration(*base);
                    &declaration.name == name
                        || declaration
                            .substitutable_by
                            .iter()
                            .any(|candidate| &schema.element_declaration(*candidate).name == name)
                }
                _ => false,
            };
            if let Some(index) = members.iter().position(matches_declaration) {
                if let Term::Element(base) = members[index].term {
                    check_declaration(schema, *element, *base, what)?;
                }
                reach[index] = Reach::Always;
                return Ok(reach);
            }
            let admitted = members.iter().position(|member| match member.term {
                Term::Wildcard(wildcard) => admits(schema, wildcard, name, &[]),
                _ => false,
            });
            let Some(index) = admitted else {
                return Err(SchemaError::new(
                    "rcase-Recurse.2",
                    format!("{what}: the base `all` group has no counterpart for `{name}`"),
                ));
            };
            reach[index] = Reach::Always;
            Ok(reach)
        }
        Term::Wildcard(wildcard) => {
            // Weak wildcards: a name the restricting wildcard can produce goes
            // to a base *element* member whenever one declares it, so that
            // member takes some of the items and the base's own wildcard
            // member can end up with none (`saxonData/Wild/wild069`).
            let mut declared = false;
            for (index, member) in members.iter().enumerate() {
                if let Term::Element(element) = member.term {
                    let name = &schema.element_declaration(*element).name;
                    if admits(schema, wildcard, name, siblings) {
                        reach[index] = Reach::Sometimes;
                        declared = true;
                    }
                }
            }
            let covering = members.iter().position(|member| match member.term {
                Term::Wildcard(base) => {
                    base.namespace.is_superset_of(&wildcard.namespace)
                        && rank(base) <= rank(wildcard)
                        && covers_disallowed(base, wildcard)
                }
                _ => false,
            });
            let Some(covering) = covering else {
                // No single base member covers the wildcard, but several
                // together may: `##local` plus `notNamespace="##local"` is
                // every namespace there is. Each of them then takes the names
                // that fall to it, and may take none (`saxonData/Wild/wild049`).
                let together: Vec<usize> = members
                    .iter()
                    .enumerate()
                    .filter(|(_, member)| match member.term {
                        Term::Wildcard(base) => {
                            base.namespace.intersects(&wildcard.namespace)
                                && rank(base) <= rank(wildcard)
                                && covers_disallowed(base, wildcard)
                        }
                        _ => false,
                    })
                    .map(|(index, _)| index)
                    .collect();
                let covered = union_covers(
                    &together
                        .iter()
                        .filter_map(|index| match members[*index].term {
                            Term::Wildcard(base) => Some(base),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                    &wildcard.namespace,
                );
                if together.len() > 1 && covered {
                    for index in together {
                        reach[index] = Reach::Sometimes;
                    }
                    return Ok(reach);
                }
                return Err(SchemaError::new(
                    "rcase-Recurse.2",
                    format!("{what}: the base `all` group has no wildcard that covers this one"),
                ));
            };
            // The covering member takes everything the base does not attribute
            // elsewhere first: to a declaration of its own, or — for a base
            // group whose wildcards overlap, which UPA no longer forbids in
            // 1.1 — to another wildcard member (`saxonData/All/all244`).
            let shared = declared
                || members.iter().enumerate().any(|(index, member)| {
                    index != covering
                        && matches!(
                            member.term,
                            Term::Wildcard(base) if base.namespace.intersects(&wildcard.namespace)
                        )
                });
            reach[covering] = if shared {
                Reach::Sometimes
            } else {
                Reach::Always
            };
            Ok(reach)
        }
        Term::Group(_) => Err(SchemaError::new(
            "rcase-Recurse.2",
            format!("{what}: a nested group cannot be mapped onto the base `all` group"),
        )),
    }
}

/// The parts of *rcase-NameAndTypeOK* that survive into XSD 1.1: when the two
/// declarations carry the same name, the restricting one must not widen the
/// base's type, nillability or fixed value.
///
/// When the names differ the restriction is using the base member's
/// substitution group, and substitutability has already established that the
/// types are compatible (`saxonData/All/all221`, `all226`).
fn check_declaration(
    schema: &Schema,
    element: crate::components::ElementId,
    base: crate::components::ElementId,
    what: &str,
) -> Result<(), SchemaError> {
    let r = schema.element_declaration(element);
    let b = schema.element_declaration(base);
    if r.name != b.name {
        return Ok(());
    }
    if r.nillable && !b.nillable {
        return Err(SchemaError::new(
            "rcase-NameAndTypeOK.2",
            format!("{what}: `{}` is nillable but its base is not", r.name),
        ));
    }
    if let Some(constraint) = &b.value_constraint
        && constraint.fixed
        && !r
            .value_constraint
            .as_ref()
            .is_some_and(|own| own.fixed && own.value == constraint.value)
    {
        return Err(SchemaError::new(
            "rcase-NameAndTypeOK.4",
            format!("{what}: `{}` is fixed in the base type", r.name),
        ));
    }
    if r.type_id != b.type_id && !schema.is_derived_from(r.type_id, b.type_id) {
        return Err(SchemaError::new(
            "rcase-NameAndTypeOK.7",
            format!(
                "{what}: the type of `{}` is not derived from its type in the base",
                r.name
            ),
        ));
    }
    Ok(())
}

/// Whether the restricting wildcard still refuses every name the base wildcard
/// refuses.
///
/// XSD 1.1 §3.10.6 *Wildcard Subset*: a wildcard is a subset of another only if
/// its `{disallowed names}` covers the other's, restricted to the names the
/// restricting wildcard's namespace constraint could admit at all
/// (`saxonData/Wild/wild051.bad`).
/// Whether the *union* of several namespace constraints admits everything
/// `target` admits.
fn union_covers(candidates: &[&Wildcard], target: &NamespaceConstraint) -> bool {
    if candidates
        .iter()
        .any(|candidate| matches!(candidate.namespace, NamespaceConstraint::Any))
    {
        return true;
    }
    let mut enumerated: BTreeSet<Option<String>> = BTreeSet::new();
    let mut excluded: Option<BTreeSet<Option<String>>> = None;
    for candidate in candidates {
        match &candidate.namespace {
            NamespaceConstraint::Any => return true,
            NamespaceConstraint::Enumeration(set) => enumerated.extend(set.iter().cloned()),
            NamespaceConstraint::Not(set) => {
                excluded = Some(match excluded {
                    // The union of `not S` and `not T` is `not (S ∩ T)`.
                    Some(previous) => previous.intersection(set).cloned().collect(),
                    None => set.clone(),
                });
            }
        }
    }
    match excluded {
        None => match target {
            NamespaceConstraint::Enumeration(wanted) => wanted.is_subset(&enumerated),
            _ => false,
        },
        Some(mut residue) => {
            residue.retain(|namespace| !enumerated.contains(namespace));
            match target {
                NamespaceConstraint::Any => residue.is_empty(),
                NamespaceConstraint::Not(wanted) => residue.is_subset(wanted),
                NamespaceConstraint::Enumeration(wanted) => {
                    wanted.iter().all(|namespace| !residue.contains(namespace))
                }
            }
        }
    }
}

pub(crate) fn covers_disallowed(base: &Wildcard, derived: &Wildcard) -> bool {
    if base.disallowed.defined && !derived.disallowed.defined {
        return false;
    }
    base.disallowed.names.iter().all(|name| {
        !derived.namespace.allows(name.namespace()) || derived.disallowed.names.contains(name)
    })
}

fn rank(wildcard: &Wildcard) -> u8 {
    match wildcard.process_contents {
        crate::components::ProcessContents::Skip => 0,
        crate::components::ProcessContents::Lax => 1,
        crate::components::ProcessContents::Strict => 2,
    }
}

fn describe(term: &Term) -> String {
    match term {
        Term::Element(_) => "an element".to_owned(),
        Term::Wildcard(_) => "a wildcard".to_owned(),
        Term::Group(_) => "a group".to_owned(),
    }
}

fn add(left: MaxOccurs, right: MaxOccurs) -> MaxOccurs {
    match (left, right) {
        (MaxOccurs::Unbounded, _) | (_, MaxOccurs::Unbounded) => MaxOccurs::Unbounded,
        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => MaxOccurs::Bounded(a.saturating_add(b)),
    }
}

fn times(left: MaxOccurs, right: MaxOccurs) -> MaxOccurs {
    match (left, right) {
        (MaxOccurs::Bounded(0), _) | (_, MaxOccurs::Bounded(0)) => MaxOccurs::Bounded(0),
        (MaxOccurs::Unbounded, _) | (_, MaxOccurs::Unbounded) => MaxOccurs::Unbounded,
        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => MaxOccurs::Bounded(a.saturating_mul(b)),
    }
}

fn larger(left: MaxOccurs, right: MaxOccurs) -> MaxOccurs {
    match (left, right) {
        (MaxOccurs::Unbounded, _) | (_, MaxOccurs::Unbounded) => MaxOccurs::Unbounded,
        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => MaxOccurs::Bounded(a.max(b)),
    }
}

fn fits(value: MaxOccurs, limit: MaxOccurs) -> bool {
    match (value, limit) {
        (_, MaxOccurs::Unbounded) => true,
        (MaxOccurs::Unbounded, MaxOccurs::Bounded(_)) => false,
        (MaxOccurs::Bounded(a), MaxOccurs::Bounded(b)) => a <= b,
    }
}
