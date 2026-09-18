//! Compiled content models: finite automata that accept child sequences.
//!
//! A `children` content model such as `(head, (p | ul)*, tail?)` is a regular
//! expression over element type names. Checking a document against it once per
//! element, by re-interpreting the expression tree, is both slow and awkward to
//! get right; this module instead compiles the model **once** into a finite
//! automaton and then walks it one child element at a time.
//!
//! The construction is Glushkov's (also called the *position automaton*):
//!
//! 1. Number the leaves of the expression — these are the automaton's
//!    non-initial states, so the NFA has exactly one state per leaf plus a
//!    virtual start, and no ε-transitions.
//! 2. Compute `nullable`, `first`, `last` and `follow` over the tree.
//! 3. Read the NFA off those sets, then determinize it by subset construction.
//!
//! Glushkov's construction is also exactly what makes XML's determinism
//! requirement checkable: Appendix E of XML 1.0 says a content model must be
//! *deterministic*, which holds precisely when no `first` or `follow` set
//! contains two distinct positions labelled with the same element type. That
//! test is [`ContentAutomaton::is_deterministic`], reported through
//! [`ValidityErrorKind::NonDeterministicContentModel`](crate::ValidityErrorKind::NonDeterministicContentModel).
//!
//! A conforming — that is, deterministic — content model determinizes to at
//! most one DFA state per position. Models that are *not* deterministic can in
//! principle blow the subset construction up exponentially, which would itself
//! be a denial-of-service surface, so the construction is capped: past the cap
//! the model is kept as an NFA and simulated. Matching results are identical
//! either way.
//!
//! # Examples
//!
//! ```
//! use oxixml_dtd::DtdParser;
//!
//! let dtd = DtdParser::new().parse_internal_subset("<!ELEMENT a (b, (c | d)*, e?)>")?;
//! let model = dtd.compiled_content("a").expect("declared");
//!
//! let mut matcher = model.matcher();
//! assert!(matcher.accept("b"));
//! assert!(matcher.accept("c"));
//! assert!(matcher.accept("d"));
//! assert!(matcher.is_final()); // `e` is optional
//! assert!(!matcher.accept("b")); // but `b` may not come back
//! # Ok::<(), oxixml_dtd::DtdError>(())
//! ```

mod glushkov;

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{ContentParticle, ContentSpec, MixedContent};

pub use glushkov::PositionAutomaton;

/// A deterministic finite automaton over element type names.
///
/// States are numbered from `0` (the start state); a transition table maps each
/// state and element type to a successor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dfa {
    transitions: Vec<BTreeMap<String, usize>>,
    accepting: Vec<bool>,
    deterministic: bool,
}

impl Dfa {
    /// The number of states.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.transitions.len()
    }

    /// Whether `state` may end the element's content.
    #[must_use]
    pub fn is_accepting(&self, state: usize) -> bool {
        self.accepting.get(state).copied().unwrap_or(false)
    }

    /// The state reached from `state` by a child element named `name`.
    #[must_use]
    pub fn step(&self, state: usize, name: &str) -> Option<usize> {
        self.transitions.get(state)?.get(name).copied()
    }

    /// The element type names accepted in `state`, sorted.
    #[must_use]
    pub fn expected(&self, state: usize) -> Vec<String> {
        self.transitions
            .get(state)
            .map(|row| row.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Whether the *source* content model already satisfied the determinism
    /// requirement of XML 1.0 Appendix E.
    ///
    /// The compiled automaton is deterministic either way — subset construction
    /// guarantees that — but a model that needed the subset construction in
    /// order to *become* deterministic is one a conforming validating processor
    /// should report.
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }
}

/// The automaton form chosen for a `children` content model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentAutomaton {
    /// The usual case: a compiled transition table.
    Table(Dfa),
    /// The fallback for models whose subset construction exceeded the internal
    /// state cap; matched by simulating the position automaton directly.
    Simulated(PositionAutomaton),
}

impl ContentAutomaton {
    /// Compiles `particle`, determinizing when that is affordable.
    #[must_use]
    pub fn compile(particle: &ContentParticle) -> Self {
        let automaton = PositionAutomaton::build(particle);
        match automaton.determinize() {
            Some(dfa) => Self::Table(dfa),
            None => Self::Simulated(automaton),
        }
    }

    /// Whether the source content model satisfies XML 1.0 Appendix E.
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        match self {
            Self::Table(dfa) => dfa.is_deterministic(),
            Self::Simulated(automaton) => automaton.is_deterministic(),
        }
    }
}

/// A ready-to-run content model.
///
/// This is [`ContentSpec`] after compilation: the models that need no automaton
/// keep their cheap representation, and only `children` models carry one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledContent {
    /// `EMPTY`: nothing at all may appear.
    Empty,
    /// `ANY`: any declared element type, plus character data.
    Any,
    /// Mixed content: character data plus, in any order and any number, the
    /// listed element types.
    Mixed(BTreeSet<String>),
    /// Element content, matched by an automaton.
    Children(ContentAutomaton),
}

impl CompiledContent {
    /// Compiles a [`ContentSpec`].
    #[must_use]
    pub fn compile(spec: &ContentSpec) -> Self {
        match spec {
            ContentSpec::Empty => Self::Empty,
            ContentSpec::Any => Self::Any,
            ContentSpec::Mixed(mixed) => Self::Mixed(compile_mixed(mixed)),
            ContentSpec::Children(particle) => Self::Children(ContentAutomaton::compile(particle)),
        }
    }

    /// Starts matching a fresh element's content.
    #[must_use]
    pub fn matcher(&self) -> ContentMatcher<'_> {
        ContentMatcher {
            model: self,
            state: MatcherState::Start,
            failed: false,
            saw_child: false,
        }
    }

    /// Whether character data (other than white space in element content) is
    /// permitted.
    #[must_use]
    pub const fn allows_character_data(&self) -> bool {
        matches!(self, Self::Any | Self::Mixed(_))
    }

    /// Whether the model satisfies XML 1.0 Appendix E determinism. Models
    /// without an automaton are trivially deterministic.
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        match self {
            Self::Children(automaton) => automaton.is_deterministic(),
            _ => true,
        }
    }
}

/// Collects the element types of a mixed content model into a set.
fn compile_mixed(mixed: &MixedContent) -> BTreeSet<String> {
    mixed.names().iter().cloned().collect()
}

/// Where a [`ContentMatcher`] currently is.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MatcherState {
    /// No child has been offered yet.
    Start,
    /// A DFA state index.
    Dfa(usize),
    /// A set of NFA positions.
    Nfa(Vec<usize>),
}

/// A cursor walking one element's children through a [`CompiledContent`].
///
/// Created by [`CompiledContent::matcher`]. Feed it each child element name
/// with [`accept`](Self::accept); when the element ends, ask
/// [`is_final`](Self::is_final) whether the content model was satisfied.
#[derive(Debug, Clone)]
pub struct ContentMatcher<'a> {
    model: &'a CompiledContent,
    state: MatcherState,
    failed: bool,
    saw_child: bool,
}

impl ContentMatcher<'_> {
    /// Offers a child element. Returns `false` when the model does not allow it
    /// here; the matcher then latches into a failed state so that one bad child
    /// does not produce a cascade of further errors.
    pub fn accept(&mut self, name: &str) -> bool {
        if self.failed {
            return false;
        }
        self.saw_child = true;
        match self.model {
            CompiledContent::Empty => {
                self.failed = true;
                false
            }
            CompiledContent::Any => true,
            CompiledContent::Mixed(names) => {
                if names.contains(name) {
                    true
                } else {
                    self.failed = true;
                    false
                }
            }
            CompiledContent::Children(ContentAutomaton::Table(dfa)) => {
                let current = match &self.state {
                    MatcherState::Dfa(state) => *state,
                    _ => 0,
                };
                match dfa.step(current, name) {
                    Some(next) => {
                        self.state = MatcherState::Dfa(next);
                        true
                    }
                    None => {
                        self.failed = true;
                        false
                    }
                }
            }
            CompiledContent::Children(ContentAutomaton::Simulated(automaton)) => {
                let next = match &self.state {
                    MatcherState::Nfa(positions) => automaton.step(positions, name),
                    _ => automaton.start_step(name),
                };
                if next.is_empty() {
                    self.failed = true;
                    false
                } else {
                    self.state = MatcherState::Nfa(next);
                    true
                }
            }
        }
    }

    /// Whether the content seen so far is a complete match.
    #[must_use]
    pub fn is_final(&self) -> bool {
        if self.failed {
            return false;
        }
        match self.model {
            CompiledContent::Empty => !self.saw_child,
            CompiledContent::Any | CompiledContent::Mixed(_) => true,
            CompiledContent::Children(ContentAutomaton::Table(dfa)) => match &self.state {
                MatcherState::Dfa(state) => dfa.is_accepting(*state),
                _ => dfa.is_accepting(0),
            },
            CompiledContent::Children(ContentAutomaton::Simulated(automaton)) => {
                match &self.state {
                    MatcherState::Nfa(positions) => automaton.accepts(positions),
                    _ => automaton.is_nullable(),
                }
            }
        }
    }

    /// Whether a child was rejected earlier.
    #[must_use]
    pub fn has_failed(&self) -> bool {
        self.failed
    }

    /// The element types that would be accepted next, sorted. Empty for `ANY`,
    /// which accepts everything.
    #[must_use]
    pub fn expected(&self) -> Vec<String> {
        match self.model {
            CompiledContent::Empty | CompiledContent::Any => Vec::new(),
            CompiledContent::Mixed(names) => names.iter().cloned().collect(),
            CompiledContent::Children(ContentAutomaton::Table(dfa)) => match &self.state {
                MatcherState::Dfa(state) => dfa.expected(*state),
                _ => dfa.expected(0),
            },
            CompiledContent::Children(ContentAutomaton::Simulated(automaton)) => {
                match &self.state {
                    MatcherState::Nfa(positions) => automaton.expected(positions),
                    _ => automaton.expected_at_start(),
                }
            }
        }
    }
}

/// Builds a [`Dfa`] from its parts. Used by the Glushkov construction.
pub(crate) fn dfa_from_parts(
    transitions: Vec<BTreeMap<String, usize>>,
    accepting: Vec<bool>,
    deterministic: bool,
) -> Dfa {
    Dfa {
        transitions,
        accepting,
        deterministic,
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
    fn sequence_of_two_requires_both_in_order() {
        let model = CompiledContent::compile(&ContentSpec::Children(ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::One), name("b", Occurrence::One)],
            occurrence: Occurrence::One,
        }));

        let mut matcher = model.matcher();
        assert!(!matcher.is_final());
        assert!(matcher.accept("a"));
        assert!(!matcher.is_final());
        assert!(matcher.accept("b"));
        assert!(matcher.is_final());

        let mut reversed = model.matcher();
        assert!(!reversed.accept("b"));
        assert!(reversed.has_failed());
    }

    #[test]
    fn star_accepts_zero_or_many() {
        let model =
            CompiledContent::compile(&ContentSpec::Children(name("a", Occurrence::ZeroOrMore)));
        assert!(model.matcher().is_final());

        let mut matcher = model.matcher();
        for _ in 0..5 {
            assert!(matcher.accept("a"));
        }
        assert!(matcher.is_final());
    }

    #[test]
    fn plus_requires_at_least_one() {
        let model =
            CompiledContent::compile(&ContentSpec::Children(name("a", Occurrence::OneOrMore)));
        assert!(!model.matcher().is_final());
        let mut matcher = model.matcher();
        assert!(matcher.accept("a"));
        assert!(matcher.is_final());
    }

    #[test]
    fn empty_rejects_any_child() {
        let model = CompiledContent::compile(&ContentSpec::Empty);
        assert!(model.matcher().is_final());
        let mut matcher = model.matcher();
        assert!(!matcher.accept("a"));
        assert!(!matcher.is_final());
    }

    #[test]
    fn mixed_content_accepts_listed_names_in_any_order() {
        let model = CompiledContent::compile(&ContentSpec::Mixed(MixedContent::new(vec![
            "a".to_owned(),
            "b".to_owned(),
        ])));
        let mut matcher = model.matcher();
        assert!(matcher.accept("b"));
        assert!(matcher.accept("a"));
        assert!(matcher.accept("b"));
        assert!(matcher.is_final());
        assert!(!matcher.accept("c"));
        assert!(model.allows_character_data());
    }

    #[test]
    fn ambiguous_models_are_flagged_but_still_match() {
        // `((b, c) | (b, d))` is the example of XML 1.0 Appendix E: on seeing
        // `b` a one-token-look-ahead processor cannot tell which branch it is
        // in. Subset construction resolves that, so matching still works.
        let model = CompiledContent::compile(&ContentSpec::Children(ContentParticle::Choice {
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
        }));
        assert!(!model.is_deterministic());

        let mut matcher = model.matcher();
        assert!(matcher.accept("b"));
        assert_eq!(matcher.expected(), ["c", "d"]);
        assert!(matcher.accept("d"));
        assert!(matcher.is_final());
        assert!(!matcher.accept("c"));
    }

    #[test]
    fn a_trailing_optional_repeat_matches_one_or_two_children() {
        let model = CompiledContent::compile(&ContentSpec::Children(ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::One), name("a", Occurrence::Optional)],
            occurrence: Occurrence::One,
        }));
        let mut matcher = model.matcher();
        assert!(matcher.accept("a"));
        assert!(matcher.is_final());
        assert!(matcher.accept("a"));
        assert!(matcher.is_final());
        assert!(!matcher.accept("a"));
    }

    #[test]
    fn deterministic_models_are_not_flagged() {
        let model = CompiledContent::compile(&ContentSpec::Children(ContentParticle::Choice {
            particles: vec![name("a", Occurrence::One), name("b", Occurrence::One)],
            occurrence: Occurrence::One,
        }));
        assert!(model.is_deterministic());
        assert_eq!(model.matcher().expected(), ["a", "b"]);
    }

    #[test]
    fn optional_prefix_lets_the_second_member_start_the_content() {
        // `(a?, b)`
        let model = CompiledContent::compile(&ContentSpec::Children(ContentParticle::Sequence {
            particles: vec![name("a", Occurrence::Optional), name("b", Occurrence::One)],
            occurrence: Occurrence::One,
        }));
        let mut direct = model.matcher();
        assert!(direct.accept("b"));
        assert!(direct.is_final());

        let mut full = model.matcher();
        assert!(full.accept("a"));
        assert!(!full.is_final());
        assert!(full.accept("b"));
        assert!(full.is_final());
    }
}
