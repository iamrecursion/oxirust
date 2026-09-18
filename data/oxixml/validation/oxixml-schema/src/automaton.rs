//! The *effective content model* of a complex type, as a finite automaton.
//!
//! A content model is compiled once, at schema-assembly time, into a
//! [Glushkov (position) automaton][glushkov]: every leaf occurrence of the
//! particle tree becomes a *position*, and the model is described by the set of
//! positions that may start it, the set that may end it, and the `follow`
//! relation between them. Validating a sequence of children is then a subset
//! simulation over those positions.
//!
//! The same automaton answers the *Unique Particle Attribution* constraint
//! ([§3.8.6](https://www.w3.org/TR/xmlschema-1/#cos-nonambig)): a content model
//! is unambiguous exactly when no reachable position set contains two positions
//! that could both match the same element, which is what
//! the schema compiler's unique-particle-attribution check tests.
//!
//! `xs:all` is not expressible as a Glushkov automaton of reasonable size, and
//! the specification restricts it to the whole of a content type anyway, so it
//! gets its own [`AllModel`] matcher.
//!
//! [glushkov]: https://en.wikipedia.org/wiki/Glushkov%27s_construction_algorithm

use std::collections::HashMap;

use crate::components::{
    Compositor, ElementId, MaxOccurs, ModelGroup, Particle, ProcessContents, Term, Wildcard,
};
use crate::error::SchemaError;
use crate::names::XsdVersion;

/// How many positions a single content model may expand to.
///
/// Only bounded `maxOccurs` unrolls, so this is reached only by deliberately
/// pathological schemas.
const POSITION_LIMIT: usize = 20_000;

/// How many optional copies a bounded `maxOccurs` may unroll into.
///
/// Beyond this the particle is compiled as if `maxOccurs` were `unbounded`: a
/// schema that writes `maxOccurs="3992"` on several sibling particles would
/// otherwise multiply out into millions of positions. The approximation only
/// ever *accepts* more content than the schema allows, and is far out of reach
/// of anything a hand-written schema does.
const UNROLL_LIMIT: u32 = 512;

/// The successively coarser unrolling budgets [`ContentModel::build`] tries.
///
/// A model whose occurrence ranges multiply out past [`POSITION_LIMIT`] is
/// rebuilt with a smaller budget rather than rejected: the *Unique Particle
/// Attribution* constraint is about telling two different **particles** apart,
/// and unrolling one particle into `n` copies never changes that — it only
/// changes how many repetitions the automaton counts. A coarser budget
/// therefore answers `cos-nonambig` exactly and only loosens the *counting*,
/// which is what `msData/particles/particlesZ033_a`, `particlesZ033_d` and
/// `particlesZ035_a` — schemas whose `maxOccurs` runs into the billions —
/// need in order to be accepted at all.
const UNROLL_BUDGETS: &[u32] = &[UNROLL_LIMIT, 64, 8, 1];

/// What a single position of the automaton matches.
#[derive(Debug, Clone)]
pub enum LeafTerm {
    /// An element declaration (matching also its substitution group).
    Element(ElementId),
    /// A wildcard.
    Wildcard(Wildcard),
}

impl LeafTerm {
    /// The `processContents` for a wildcard position.
    #[must_use]
    pub fn process_contents(&self) -> Option<ProcessContents> {
        match self {
            Self::Element(_) => None,
            Self::Wildcard(wildcard) => Some(wildcard.process_contents),
        }
    }
}

/// Flattens the members of an `xs:all` group into `items`.
///
/// Under XSD 1.1 a member may itself be an `all` group — the schema wrote
/// `<xs:group ref="…"/>` and the referenced group's compositor is `all` — in
/// which case its members join the enclosing group, their occurrence ranges
/// multiplied by the reference's own. XSD 1.0 forbids this outright
/// (*cos-all-limited.2*).
fn collect_all_items(
    group: &ModelGroup,
    version: XsdVersion,
    items: &mut Vec<AllItem>,
) -> Result<(), SchemaError> {
    for child in &group.particles {
        match &child.term {
            Term::Element(id) => items.push(AllItem {
                term: LeafTerm::Element(*id),
                min_occurs: child.min_occurs,
                max_occurs: child.max_occurs,
            }),
            Term::Wildcard(wildcard) => items.push(AllItem {
                term: LeafTerm::Wildcard(wildcard.clone()),
                min_occurs: child.min_occurs,
                max_occurs: child.max_occurs,
            }),
            Term::Group(nested)
                if version.is_1_1()
                    && nested.compositor == Compositor::All
                    && child.min_occurs == 1
                    && child.max_occurs == MaxOccurs::Bounded(1) =>
            {
                collect_all_items(nested, version, items)?;
            }
            Term::Group(_) => {
                return Err(SchemaError::new(
                    "cos-all-limited.2",
                    "the children of an `all` group must be element declarations or wildcards",
                ));
            }
        }
    }
    Ok(())
}

/// One item of an `xs:all` group.
#[derive(Debug, Clone)]
pub struct AllItem {
    /// What it matches.
    pub term: LeafTerm,
    /// `minOccurs` (0 or 1 in XSD 1.0).
    pub min_occurs: u32,
    /// `maxOccurs` (1 in XSD 1.0, any value in 1.1).
    pub max_occurs: MaxOccurs,
}

/// The matcher for an `xs:all` content model.
#[derive(Debug, Clone)]
pub struct AllModel {
    /// The items, in the order they were declared.
    pub items: Vec<AllItem>,
    /// Whether the whole group may be absent (`minOccurs="0"` on the group).
    pub emptiable: bool,
}

/// The Glushkov automaton of a sequence/choice content model.
#[derive(Debug, Clone)]
pub struct Fsm {
    /// The terms, indexed by position.
    pub positions: Vec<LeafTerm>,
    /// Which *particle* of the schema each position was unrolled from.
    ///
    /// `maxOccurs="5"` turns one particle into five positions; Unique Particle
    /// Attribution is about telling two **different** particles apart, so two
    /// positions that share a particle never compete.
    pub particles: Vec<usize>,
    /// The positions that may start the model.
    pub first: Vec<usize>,
    /// Whether each position may end the model.
    pub last: Vec<bool>,
    /// The positions that may follow each position.
    pub follow: Vec<Vec<usize>>,
    /// Whether the model accepts the empty sequence.
    pub nullable: bool,
}

/// A compiled content model.
#[derive(Debug, Clone)]
pub enum ContentModel {
    /// No element children are allowed.
    Empty,
    /// An `xs:all` group.
    All(AllModel),
    /// Everything else.
    Fsm(Fsm),
}

/// Whether a term can match the empty sequence.
fn term_is_emptiable(term: &Term) -> bool {
    match term {
        Term::Element(_) | Term::Wildcard(_) => false,
        Term::Group(group) => match group.compositor {
            // A `choice` with no branches accepts nothing at all — not even
            // the empty sequence — so it is *not* ·emptiable· (§3.9.6.3).
            Compositor::Choice => group.particles.iter().any(particle_is_emptiable),
            Compositor::Sequence | Compositor::All => {
                group.particles.iter().all(particle_is_emptiable)
            }
        },
    }
}

fn particle_is_emptiable(particle: &Particle) -> bool {
    particle.min_occurs == 0 || term_is_emptiable(&particle.term)
}

/// The three sets Glushkov's construction computes bottom-up.
#[derive(Debug, Default)]
struct Fragment {
    nullable: bool,
    first: Vec<usize>,
    last: Vec<usize>,
}

struct Builder {
    positions: Vec<LeafTerm>,
    particles: Vec<usize>,
    follow: Vec<Vec<usize>>,
    /// Identity of each leaf term of the particle tree, keyed by its address.
    /// The tree is immutable while it is compiled, so the address is a stable
    /// name for "the same particle".
    identities: HashMap<usize, usize>,
    /// How many copies a bounded occurrence range may unroll into.
    unroll: u32,
}

impl Builder {
    fn new(unroll: u32) -> Self {
        Self {
            positions: Vec::new(),
            particles: Vec::new(),
            follow: Vec::new(),
            identities: HashMap::new(),
            unroll,
        }
    }

    fn identity(&mut self, term: &Term) -> usize {
        let key = std::ptr::from_ref(term) as usize;
        let next = self.identities.len();
        *self.identities.entry(key).or_insert(next)
    }

    fn add_position(&mut self, term: LeafTerm, particle: usize) -> Result<usize, SchemaError> {
        if self.positions.len() >= POSITION_LIMIT {
            return Err(SchemaError::new(
                "cos-nonambig",
                "the content model expands to more positions than this implementation supports; \
                 reduce the occurrence ranges",
            ));
        }
        self.positions.push(term);
        self.particles.push(particle);
        self.follow.push(Vec::new());
        Ok(self.positions.len() - 1)
    }

    fn link(&mut self, from: &[usize], to: &[usize]) {
        for &source in from {
            for &target in to {
                if !self.follow[source].contains(&target) {
                    self.follow[source].push(target);
                }
            }
        }
    }

    /// Concatenates `left` and `right` into `left`.
    fn concat(&mut self, left: Fragment, right: Fragment) -> Fragment {
        self.link(&left.last, &right.first);
        let mut first = left.first;
        if left.nullable {
            for position in &right.first {
                if !first.contains(position) {
                    first.push(*position);
                }
            }
        }
        let mut last = right.last;
        if right.nullable {
            for position in &left.last {
                if !last.contains(position) {
                    last.push(*position);
                }
            }
        }
        Fragment {
            nullable: left.nullable && right.nullable,
            first,
            last,
        }
    }

    fn union(left: Fragment, right: Fragment) -> Fragment {
        let mut first = left.first;
        for position in right.first {
            if !first.contains(&position) {
                first.push(position);
            }
        }
        let mut last = left.last;
        for position in right.last {
            if !last.contains(&position) {
                last.push(position);
            }
        }
        Fragment {
            nullable: left.nullable || right.nullable,
            first,
            last,
        }
    }

    fn empty() -> Fragment {
        Fragment {
            nullable: true,
            first: Vec::new(),
            last: Vec::new(),
        }
    }

    fn nothing() -> Fragment {
        Fragment {
            nullable: false,
            first: Vec::new(),
            last: Vec::new(),
        }
    }

    /// Builds one *unrolled* copy of the particle's term.
    fn term_fragment(&mut self, term: &Term) -> Result<Fragment, SchemaError> {
        Ok(match term {
            Term::Element(id) => {
                let particle = self.identity(term);
                let position = self.add_position(LeafTerm::Element(*id), particle)?;
                Fragment {
                    nullable: false,
                    first: vec![position],
                    last: vec![position],
                }
            }
            Term::Wildcard(wildcard) => {
                let particle = self.identity(term);
                let position = self.add_position(LeafTerm::Wildcard(wildcard.clone()), particle)?;
                Fragment {
                    nullable: false,
                    first: vec![position],
                    last: vec![position],
                }
            }
            Term::Group(group) => self.group_fragment(group)?,
        })
    }

    fn group_fragment(&mut self, group: &ModelGroup) -> Result<Fragment, SchemaError> {
        match group.compositor {
            Compositor::Sequence => {
                let mut accumulated = Self::empty();
                for particle in &group.particles {
                    let next = self.particle_fragment(particle)?;
                    accumulated = self.concat(accumulated, next);
                }
                Ok(accumulated)
            }
            Compositor::Choice => {
                if group.particles.is_empty() {
                    // An empty `choice` matches nothing at all.
                    return Ok(Self::nothing());
                }
                let mut accumulated = Self::nothing();
                for particle in &group.particles {
                    let next = self.particle_fragment(particle)?;
                    accumulated = Self::union(accumulated, next);
                }
                Ok(accumulated)
            }
            Compositor::All => Err(SchemaError::new(
                "cos-all-limited",
                "an `all` model group may only be the whole content model of a complex type",
            )),
        }
    }

    fn particle_fragment(&mut self, particle: &Particle) -> Result<Fragment, SchemaError> {
        if particle.max_occurs.is_zero() {
            return Ok(Self::empty());
        }
        let min = particle.min_occurs;
        match particle.max_occurs {
            MaxOccurs::Unbounded => {
                // `x{0,}` is `x*`, and so is `x{n,}` when `x` itself can match
                // the empty sequence; otherwise `x{n,}` is `x` repeated `n-1`
                // times followed by `x+`.
                if min == 0 || term_is_emptiable(&particle.term) {
                    let body = self.term_fragment(&particle.term)?;
                    return Ok(self.star(body));
                }
                let mut accumulated = Self::empty();
                for _ in 0..min.min(self.unroll).max(1) - 1 {
                    let next = self.term_fragment(&particle.term)?;
                    accumulated = self.concat(accumulated, next);
                }
                let tail = self.term_fragment(&particle.term)?;
                let tail = self.plus(tail);
                Ok(self.concat(accumulated, tail))
            }
            MaxOccurs::Bounded(max) if max.saturating_sub(min) > self.unroll => {
                // Too wide to unroll; see `UNROLL_LIMIT`.
                self.particle_fragment(&Particle {
                    min_occurs: min,
                    max_occurs: MaxOccurs::Unbounded,
                    term: particle.term.clone(),
                })
            }
            MaxOccurs::Bounded(max) => {
                let min = min.min(self.unroll);
                let mut accumulated = Self::empty();
                for _ in 0..min {
                    let next = self.term_fragment(&particle.term)?;
                    accumulated = self.concat(accumulated, next);
                }
                // The optional tail is built right-to-left so that
                // `x{0,3}` becomes `(x (x (x)?)?)?`.
                let optional_count = max.saturating_sub(min);
                let mut tail = Self::empty();
                for _ in 0..optional_count {
                    let head = self.term_fragment(&particle.term)?;
                    let joined = self.concat(head, tail);
                    tail = Fragment {
                        nullable: true,
                        first: joined.first,
                        last: joined.last,
                    };
                }
                Ok(self.concat(accumulated, tail))
            }
        }
    }

    /// `x*`: loop back, and accept the empty sequence.
    fn star(&mut self, fragment: Fragment) -> Fragment {
        self.link(&fragment.last, &fragment.first);
        Fragment {
            nullable: true,
            first: fragment.first,
            last: fragment.last,
        }
    }

    /// `x+`: loop back, keeping the fragment's own nullability.
    ///
    /// Building `x{1,}` as `x x*` — two copies — would make every content model
    /// whose term can match the empty sequence look ambiguous, because the two
    /// copies' start positions would compete. One looping copy is both smaller
    /// and faithful.
    fn plus(&mut self, fragment: Fragment) -> Fragment {
        self.link(&fragment.last, &fragment.first);
        fragment
    }
}

impl ContentModel {
    /// Compiles the effective content model of `particle`.
    ///
    /// # Errors
    ///
    /// Returns `cos-all-limited` when an `xs:all` group appears anywhere other
    /// than as the whole content model, and `cos-nonambig` when the model
    /// expands past this implementation's position limit.
    pub fn build(particle: &Particle) -> Result<Self, SchemaError> {
        Self::build_for(particle, XsdVersion::V1_0)
    }

    /// The same, under a named version of XML Schema.
    ///
    /// XSD 1.1 §3.8.6 relaxes *All Group Limited*: an `xs:all` group may hold a
    /// reference to a named model group whose compositor is `all`, whose
    /// members are merged into the enclosing group
    /// (`saxonData/All/all007`, `all206`).
    ///
    /// # Errors
    ///
    /// The same as [`build`](Self::build).
    pub fn build_for(particle: &Particle, version: XsdVersion) -> Result<Self, SchemaError> {
        if let Term::Group(group) = &particle.term
            && group.compositor == Compositor::All
        {
            if particle.min_occurs > 1 || !matches!(particle.max_occurs, MaxOccurs::Bounded(1)) {
                return Err(SchemaError::new(
                    "cos-all-limited.1",
                    "an `all` group must have `minOccurs` 0 or 1 and `maxOccurs` 1",
                ));
            }
            let mut items = Vec::with_capacity(group.particles.len());
            collect_all_items(group, version, &mut items)?;
            return Ok(Self::All(AllModel {
                items,
                emptiable: particle.min_occurs == 0,
            }));
        }

        let mut outcome = None;
        for budget in UNROLL_BUDGETS {
            let mut builder = Builder::new(*budget);
            match builder.particle_fragment(particle) {
                Ok(fragment) => {
                    outcome = Some((builder, fragment));
                    break;
                }
                // Only the position limit is worth retrying with a coarser
                // budget; every other failure is a property of the model.
                Err(error) if error.code() == "cos-nonambig" => continue,
                Err(error) => return Err(error),
            }
        }
        let (builder, fragment) = outcome.ok_or_else(|| {
            SchemaError::new(
                "cos-nonambig",
                "the content model expands to more positions than this implementation supports; \
                 reduce the occurrence ranges",
            )
        })?;
        if builder.positions.is_empty() {
            // A model with no leaves normally accepts exactly the empty
            // sequence. An `xs:choice` with no branches is the exception: it
            // accepts the *empty language*, so even an empty element is
            // invalid against it (`saxonData/Complex/complex022`). §3.4.2.3.3
            // clause 2.1.3 only lets such a choice count as ·empty content·
            // when its `minOccurs` is 0, so the distinction has to survive
            // here.
            if fragment.nullable {
                return Ok(Self::Empty);
            }
            return Ok(Self::Fsm(Fsm {
                positions: Vec::new(),
                particles: Vec::new(),
                first: Vec::new(),
                last: Vec::new(),
                follow: Vec::new(),
                nullable: false,
            }));
        }
        let mut last = vec![false; builder.positions.len()];
        for position in fragment.last {
            last[position] = true;
        }
        Ok(Self::Fsm(Fsm {
            positions: builder.positions,
            particles: builder.particles,
            first: fragment.first,
            last,
            follow: builder.follow,
            nullable: fragment.nullable,
        }))
    }

    /// Whether the model accepts an empty child sequence.
    #[must_use]
    pub fn accepts_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::All(model) => {
                model.emptiable || model.items.iter().all(|item| item.min_occurs == 0)
            }
            Self::Fsm(fsm) => fsm.nullable,
        }
    }

    /// The position sets a Unique Particle Attribution check must inspect: the
    /// start set and every `follow` set.
    #[must_use]
    pub fn competing_sets(&self) -> Vec<&[usize]> {
        match self {
            Self::Empty | Self::All(_) => Vec::new(),
            Self::Fsm(fsm) => {
                let mut sets: Vec<&[usize]> = Vec::with_capacity(fsm.follow.len() + 1);
                sets.push(&fsm.first);
                for set in &fsm.follow {
                    sets.push(set);
                }
                sets
            }
        }
    }

    /// The term at `position`, for a set returned by
    /// [`competing_sets`](Self::competing_sets).
    #[must_use]
    pub fn position_term(&self, position: usize) -> Option<&LeafTerm> {
        match self {
            Self::Fsm(fsm) => fsm.positions.get(position),
            _ => None,
        }
    }

    /// The particle `position` was unrolled from.
    #[must_use]
    pub fn position_particle(&self, position: usize) -> Option<usize> {
        match self {
            Self::Fsm(fsm) => fsm.particles.get(position).copied(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{ModelGroup, NamespaceConstraint};

    fn element(index: u32, min: u32, max: MaxOccurs) -> Particle {
        Particle {
            min_occurs: min,
            max_occurs: max,
            term: Term::Element(ElementId(index)),
        }
    }

    fn group(compositor: Compositor, particles: Vec<Particle>) -> Particle {
        Particle {
            min_occurs: 1,
            max_occurs: MaxOccurs::Bounded(1),
            term: Term::Group(ModelGroup {
                compositor,
                particles,
            }),
        }
    }

    #[test]
    fn a_sequence_links_positions() -> Result<(), SchemaError> {
        let particle = group(
            Compositor::Sequence,
            vec![
                element(0, 1, MaxOccurs::Bounded(1)),
                element(1, 1, MaxOccurs::Bounded(1)),
            ],
        );
        let ContentModel::Fsm(fsm) = ContentModel::build(&particle)? else {
            panic!("expected a finite-state model");
        };
        assert_eq!(fsm.first, vec![0]);
        assert_eq!(fsm.follow[0], vec![1]);
        assert!(fsm.last[1] && !fsm.last[0]);
        assert!(!fsm.nullable);
        Ok(())
    }

    #[test]
    fn bounded_repetition_unrolls_exactly() -> Result<(), SchemaError> {
        let particle = group(
            Compositor::Sequence,
            vec![element(0, 2, MaxOccurs::Bounded(3))],
        );
        let ContentModel::Fsm(fsm) = ContentModel::build(&particle)? else {
            panic!("expected a finite-state model");
        };
        assert_eq!(fsm.positions.len(), 3);
        assert!(!fsm.nullable);
        assert!(fsm.last[1] && fsm.last[2]);
        Ok(())
    }

    #[test]
    fn unbounded_repetition_loops() -> Result<(), SchemaError> {
        let particle = group(
            Compositor::Sequence,
            vec![element(0, 0, MaxOccurs::Unbounded)],
        );
        let ContentModel::Fsm(fsm) = ContentModel::build(&particle)? else {
            panic!("expected a finite-state model");
        };
        assert_eq!(fsm.positions.len(), 1);
        assert!(fsm.nullable);
        assert_eq!(fsm.follow[0], vec![0]);
        Ok(())
    }

    #[test]
    fn an_all_group_becomes_its_own_model() -> Result<(), SchemaError> {
        let particle = group(
            Compositor::All,
            vec![
                element(0, 1, MaxOccurs::Bounded(1)),
                element(1, 0, MaxOccurs::Bounded(1)),
            ],
        );
        let ContentModel::All(model) = ContentModel::build(&particle)? else {
            panic!("expected an all model");
        };
        assert_eq!(model.items.len(), 2);
        assert!(!model.emptiable);
        Ok(())
    }

    #[test]
    fn a_nested_all_group_is_rejected() {
        let particle = group(
            Compositor::Sequence,
            vec![group(
                Compositor::All,
                vec![element(0, 1, MaxOccurs::Bounded(1))],
            )],
        );
        assert!(ContentModel::build(&particle).is_err());
    }

    #[test]
    fn a_wildcard_becomes_a_position() -> Result<(), SchemaError> {
        let particle = group(
            Compositor::Sequence,
            vec![Particle {
                min_occurs: 1,
                max_occurs: MaxOccurs::Bounded(1),
                term: Term::Wildcard(Wildcard {
                    namespace: NamespaceConstraint::Any,
                    process_contents: ProcessContents::Lax,
                    disallowed: crate::components::DisallowedNames::default(),
                }),
            }],
        );
        let ContentModel::Fsm(fsm) = ContentModel::build(&particle)? else {
            panic!("expected a finite-state model");
        };
        assert!(matches!(fsm.positions[0], LeafTerm::Wildcard(_)));
        Ok(())
    }
}
