//! Glushkov (position) automaton construction for `children` content models.
//!
//! Every leaf of the content model gets a *position*; the automaton's states
//! are those positions plus a virtual start. Because the construction is
//! ε-free, the sets it computes (`first`, `last`, `follow`) are exactly what
//! XML 1.0 Appendix E needs in order to decide whether a content model is
//! deterministic: it is, precisely when no such set holds two distinct
//! positions carrying the same element type name.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::ContentParticle;

/// The largest DFA the subset construction will build before giving up and
/// leaving the model to be matched by NFA simulation.
///
/// Conforming (deterministic) content models compile to at most one state per
/// position, so this only ever bites on models that already violate XML 1.0
/// Appendix E — where an unbounded subset construction would be a denial-of-
/// service surface.
const MAX_DFA_STATES: usize = 1024;

/// The ε-free position automaton of a content model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionAutomaton {
    symbols: Vec<String>,
    first: BTreeSet<usize>,
    last: BTreeSet<usize>,
    follow: Vec<BTreeSet<usize>>,
    nullable: bool,
}

impl PositionAutomaton {
    /// Builds the position automaton of `particle`.
    #[must_use]
    pub fn build(particle: &ContentParticle) -> Self {
        let mut builder = Builder {
            symbols: Vec::new(),
            follow: Vec::new(),
        };
        let fragment = builder.visit(particle);
        Self {
            symbols: builder.symbols,
            first: fragment.first,
            last: fragment.last,
            follow: builder.follow,
            nullable: fragment.nullable,
        }
    }

    /// The number of leaves in the content model.
    #[must_use]
    pub fn position_count(&self) -> usize {
        self.symbols.len()
    }

    /// The element type name carried by `position`.
    #[must_use]
    pub fn symbol(&self, position: usize) -> Option<&str> {
        self.symbols.get(position).map(String::as_str)
    }

    /// Whether the content model accepts the empty sequence.
    #[must_use]
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Whether the model satisfies the determinism requirement of XML 1.0
    /// Appendix E.
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        if !self.is_unambiguous(&self.first) {
            return false;
        }
        self.follow.iter().all(|set| self.is_unambiguous(set))
    }

    /// Whether `set` holds no two distinct positions with the same symbol.
    fn is_unambiguous(&self, set: &BTreeSet<usize>) -> bool {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for &position in set {
            let Some(symbol) = self.symbol(position) else {
                return false;
            };
            if !seen.insert(symbol) {
                return false;
            }
        }
        true
    }

    /// The positions reachable from the start by a child named `name`.
    #[must_use]
    pub fn start_step(&self, name: &str) -> Vec<usize> {
        self.first
            .iter()
            .copied()
            .filter(|&position| self.symbol(position) == Some(name))
            .collect()
    }

    /// The positions reachable from `states` by a child named `name`.
    #[must_use]
    pub fn step(&self, states: &[usize], name: &str) -> Vec<usize> {
        let mut next = BTreeSet::new();
        for &state in states {
            let Some(follow) = self.follow.get(state) else {
                continue;
            };
            for &candidate in follow {
                if self.symbol(candidate) == Some(name) {
                    next.insert(candidate);
                }
            }
        }
        next.into_iter().collect()
    }

    /// Whether `states` may end the element's content.
    #[must_use]
    pub fn accepts(&self, states: &[usize]) -> bool {
        states.iter().any(|state| self.last.contains(state))
    }

    /// The element type names accepted from `states`, sorted.
    #[must_use]
    pub fn expected(&self, states: &[usize]) -> Vec<String> {
        let mut names = BTreeSet::new();
        for &state in states {
            let Some(follow) = self.follow.get(state) else {
                continue;
            };
            for &candidate in follow {
                if let Some(symbol) = self.symbol(candidate) {
                    names.insert(symbol.to_owned());
                }
            }
        }
        names.into_iter().collect()
    }

    /// The element type names accepted from the start state, sorted.
    #[must_use]
    pub fn expected_at_start(&self) -> Vec<String> {
        self.first
            .iter()
            .filter_map(|&position| self.symbol(position))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Determinizes by subset construction, or returns `None` when the DFA
    /// would exceed the internal cap of `MAX_DFA_STATES` (1024) states.
    #[must_use]
    pub fn determinize(&self) -> Option<super::Dfa> {
        let deterministic = self.is_deterministic();
        let mut index: BTreeMap<(bool, Vec<usize>), usize> = BTreeMap::new();
        let mut queue: Vec<(bool, Vec<usize>)> = Vec::new();
        let mut transitions: Vec<BTreeMap<String, usize>> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();

        let start = (true, Vec::new());
        index.insert(start.clone(), 0);
        queue.push(start);
        transitions.push(BTreeMap::new());
        accepting.push(self.nullable);

        let mut cursor = 0;
        while cursor < queue.len() {
            let (is_start, states) = queue[cursor].clone();
            let alphabet: BTreeSet<&str> = if is_start {
                self.first.iter().filter_map(|&p| self.symbol(p)).collect()
            } else {
                states
                    .iter()
                    .filter_map(|&state| self.follow.get(state))
                    .flatten()
                    .filter_map(|&p| self.symbol(p))
                    .collect()
            };
            for symbol in alphabet {
                let next = if is_start {
                    self.start_step(symbol)
                } else {
                    self.step(&states, symbol)
                };
                if next.is_empty() {
                    continue;
                }
                let key = (false, next.clone());
                let target = match index.get(&key) {
                    Some(&existing) => existing,
                    None => {
                        if transitions.len() >= MAX_DFA_STATES {
                            return None;
                        }
                        let id = transitions.len();
                        index.insert(key.clone(), id);
                        queue.push(key);
                        transitions.push(BTreeMap::new());
                        accepting.push(self.accepts(&next));
                        id
                    }
                };
                transitions[cursor].insert(symbol.to_owned(), target);
            }
            cursor += 1;
        }

        Some(super::dfa_from_parts(transitions, accepting, deterministic))
    }
}

/// The `nullable`/`first`/`last` triple of one sub-expression.
struct Fragment {
    nullable: bool,
    first: BTreeSet<usize>,
    last: BTreeSet<usize>,
}

/// Accumulates positions and `follow` sets while walking the content model.
struct Builder {
    symbols: Vec<String>,
    follow: Vec<BTreeSet<usize>>,
}

impl Builder {
    /// Allocates a position for `name`.
    fn position(&mut self, name: &str) -> usize {
        let id = self.symbols.len();
        self.symbols.push(name.to_owned());
        self.follow.push(BTreeSet::new());
        id
    }

    /// Adds `targets` to the follow set of every position in `sources`.
    fn add_follow(&mut self, sources: &BTreeSet<usize>, targets: &BTreeSet<usize>) {
        for &source in sources {
            if let Some(set) = self.follow.get_mut(source) {
                set.extend(targets.iter().copied());
            }
        }
    }

    /// Computes the fragment of `particle`, including its occurrence operator.
    fn visit(&mut self, particle: &ContentParticle) -> Fragment {
        let (mut fragment, occurrence) = match particle {
            ContentParticle::Name { name, occurrence } => {
                let position = self.position(name);
                let mut set = BTreeSet::new();
                set.insert(position);
                (
                    Fragment {
                        nullable: false,
                        first: set.clone(),
                        last: set,
                    },
                    *occurrence,
                )
            }
            ContentParticle::Choice {
                particles,
                occurrence,
            } => (self.visit_choice(particles), *occurrence),
            ContentParticle::Sequence {
                particles,
                occurrence,
            } => (self.visit_sequence(particles), *occurrence),
        };

        if occurrence.allows_many() {
            let first = fragment.first.clone();
            let last = fragment.last.clone();
            self.add_follow(&last, &first);
        }
        if occurrence.allows_zero() {
            fragment.nullable = true;
        }
        fragment
    }

    /// `first`/`last`/`nullable` of an alternation.
    fn visit_choice(&mut self, particles: &[ContentParticle]) -> Fragment {
        let mut nullable = false;
        let mut first = BTreeSet::new();
        let mut last = BTreeSet::new();
        for particle in particles {
            let fragment = self.visit(particle);
            nullable |= fragment.nullable;
            first.extend(fragment.first);
            last.extend(fragment.last);
        }
        Fragment {
            nullable,
            first,
            last,
        }
    }

    /// `first`/`last`/`nullable` of a concatenation, wiring up the `follow`
    /// sets that join consecutive members.
    ///
    /// Written iteratively over the standard two-operand rules
    /// `first(e1 e2) = first(e1) ∪ (nullable(e1) ? first(e2) : ∅)` and
    /// `last(e1 e2)  = last(e2)  ∪ (nullable(e2) ? last(e1)  : ∅)`.
    fn visit_sequence(&mut self, particles: &[ContentParticle]) -> Fragment {
        let mut nullable = true;
        let mut first = BTreeSet::new();
        let mut last: BTreeSet<usize> = BTreeSet::new();

        for particle in particles {
            let fragment = self.visit(particle);

            if nullable {
                first.extend(fragment.first.iter().copied());
            }
            self.add_follow(&last, &fragment.first);

            if fragment.nullable {
                last.extend(fragment.last.iter().copied());
            } else {
                last = fragment.last;
            }
            nullable &= fragment.nullable;
        }

        Fragment {
            nullable,
            first,
            last,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Occurrence;

    fn name(text: &str, occurrence: Occurrence) -> ContentParticle {
        ContentParticle::Name {
            name: text.to_owned(),
            occurrence,
        }
    }

    #[test]
    fn sequence_follow_sets_link_consecutive_members() {
        let particle = ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::One), name("b", Occurrence::One)],
            occurrence: Occurrence::One,
        };
        let automaton = PositionAutomaton::build(&particle);
        assert_eq!(automaton.position_count(), 2);
        assert!(!automaton.is_nullable());
        assert_eq!(automaton.start_step("a"), [0]);
        assert_eq!(automaton.step(&[0], "b"), [1]);
        assert!(automaton.accepts(&[1]));
        assert!(!automaton.accepts(&[0]));
        assert!(automaton.is_deterministic());
    }

    #[test]
    fn optional_members_make_later_members_reachable_from_the_start() {
        // `(a?, b)` — `b` is in `first` because `a?` is nullable.
        let particle = ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::Optional), name("b", Occurrence::One)],
            occurrence: Occurrence::One,
        };
        let automaton = PositionAutomaton::build(&particle);
        assert_eq!(automaton.expected_at_start(), ["a", "b"]);
        assert!(!automaton.is_nullable());
    }

    #[test]
    fn trailing_optional_members_stay_in_last() {
        // `(a, b?)` — the content may end after `a`.
        let particle = ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::One), name("b", Occurrence::Optional)],
            occurrence: Occurrence::One,
        };
        let automaton = PositionAutomaton::build(&particle);
        assert!(automaton.accepts(&[0]));
        assert!(automaton.accepts(&[1]));
    }

    #[test]
    fn the_appendix_e_example_is_ambiguous() {
        // `((b, c) | (b, d))` is the example XML 1.0 Appendix E itself gives:
        // on seeing `b` the processor cannot tell which branch it entered.
        let particle = ContentParticle::Choice {
            particles: vec![
                ContentParticle::Sequence {
                    particles: vec![name("b", Occurrence::One), name("c", Occurrence::One)],
                    occurrence: Occurrence::One,
                },
                ContentParticle::Sequence {
                    particles: vec![name("b", Occurrence::One), name("d", Occurrence::One)],
                    occurrence: Occurrence::One,
                },
            ],
            occurrence: Occurrence::One,
        };
        assert!(!PositionAutomaton::build(&particle).is_deterministic());
    }

    #[test]
    fn an_optional_leading_repeat_is_ambiguous() {
        // `(a?, a)` — the first `a` could match either position.
        let particle = ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::Optional), name("a", Occurrence::One)],
            occurrence: Occurrence::One,
        };
        assert!(!PositionAutomaton::build(&particle).is_deterministic());
    }

    #[test]
    fn a_trailing_optional_repeat_stays_deterministic() {
        // `(a, a?)` on the other hand is unambiguous: the first `a` can only be
        // the first position, so no look-ahead is ever needed.
        let particle = ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::One), name("a", Occurrence::Optional)],
            occurrence: Occurrence::One,
        };
        assert!(PositionAutomaton::build(&particle).is_deterministic());
    }

    #[test]
    fn nested_star_loops_back_to_its_own_first_set() {
        // `(a, b)*`
        let particle = ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::One), name("b", Occurrence::One)],
            occurrence: Occurrence::ZeroOrMore,
        };
        let automaton = PositionAutomaton::build(&particle);
        assert!(automaton.is_nullable());
        let after_a = automaton.start_step("a");
        let after_b = automaton.step(&after_a, "b");
        assert!(automaton.accepts(&after_b));
        assert_eq!(automaton.step(&after_b, "a"), after_a);
    }
}
