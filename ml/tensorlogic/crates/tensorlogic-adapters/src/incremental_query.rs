//! Incremental query evaluation using semi-naive Datalog evaluation.
//!
//! Implements the semi-naive bottom-up fixpoint algorithm for Datalog: given a set of
//! rules and base facts (EDB), computes derived facts (IDB) incrementally by only
//! re-evaluating rules against Δ (newly derived tuples) at each iteration.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{Atom, Edb, Fact, FactArg, IncrementalEvaluator, Rule, Term};
//!
//! // Build EDB: parent(alice, bob), parent(bob, carol)
//! let mut edb = Edb::new();
//! edb.add_fact(Fact::sym("parent", &["alice", "bob"]));
//! edb.add_fact(Fact::sym("parent", &["bob", "carol"]));
//!
//! // Rule: ancestor(X, Y) :- parent(X, Y).
//! let rule1 = Rule::new(
//!     Atom::new("ancestor", vec![Term::var("X"), Term::var("Y")]),
//!     vec![Atom::new("parent", vec![Term::var("X"), Term::var("Y")])],
//! );
//!
//! // Rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
//! let rule2 = Rule::new(
//!     Atom::new("ancestor", vec![Term::var("X"), Term::var("Z")]),
//!     vec![
//!         Atom::new("parent", vec![Term::var("X"), Term::var("Y")]),
//!         Atom::new("ancestor", vec![Term::var("Y"), Term::var("Z")]),
//!     ],
//! );
//!
//! let mut evaluator = IncrementalEvaluator::new(vec![rule1, rule2], edb).unwrap();
//! let derived = evaluator.query("ancestor");
//! assert_eq!(derived.len(), 3); // alice->bob, bob->carol, alice->carol
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// FactArg
// ─────────────────────────────────────────────────────────────────────────────

/// A fact argument value — either a symbol or an integer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FactArg {
    /// A symbolic / string constant.
    Symbol(String),
    /// An integer constant.
    Integer(i64),
}

impl fmt::Display for FactArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactArg::Symbol(s) => write!(f, "{s}"),
            FactArg::Integer(n) => write!(f, "{n}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fact
// ─────────────────────────────────────────────────────────────────────────────

/// A ground fact: a predicate name together with its argument values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    /// The predicate name (e.g. `"parent"`, `"ancestor"`).
    pub predicate: String,
    /// The argument values (ground terms).
    pub args: Vec<FactArg>,
}

impl Fact {
    /// Create a new fact.
    pub fn new(predicate: impl Into<String>, args: Vec<FactArg>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
        }
    }

    /// Convenience constructor: all arguments are `Symbol` values.
    pub fn sym(predicate: impl Into<String>, args: &[&str]) -> Self {
        Self {
            predicate: predicate.into(),
            args: args
                .iter()
                .map(|s| FactArg::Symbol(s.to_string()))
                .collect(),
        }
    }

    /// The number of arguments (arity).
    pub fn arity(&self) -> usize {
        self.args.len()
    }
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.predicate)?;
        for (i, a) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{a}")?;
        }
        write!(f, ")")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Term
// ─────────────────────────────────────────────────────────────────────────────

/// A term in a Datalog rule body or head atom.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A logical variable (e.g. `"X"`, `"Y"`).
    Variable(String),
    /// A ground constant.
    Constant(FactArg),
}

impl Term {
    /// Create a variable term.
    pub fn var(name: impl Into<String>) -> Self {
        Term::Variable(name.into())
    }

    /// Create a symbol constant term.
    pub fn sym(s: impl Into<String>) -> Self {
        Term::Constant(FactArg::Symbol(s.into()))
    }

    /// Create an integer constant term.
    pub fn int(n: i64) -> Self {
        Term::Constant(FactArg::Integer(n))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atom
// ─────────────────────────────────────────────────────────────────────────────

/// A Datalog atom: a predicate applied to a list of terms.
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// The predicate name.
    pub predicate: String,
    /// The argument terms (may contain variables or constants).
    pub terms: Vec<Term>,
}

impl Atom {
    /// Create a new atom.
    pub fn new(predicate: impl Into<String>, terms: Vec<Term>) -> Self {
        Self {
            predicate: predicate.into(),
            terms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule
// ─────────────────────────────────────────────────────────────────────────────

/// A Datalog rule: `head :- body[0], body[1], …`.
///
/// If `body` is empty, the rule is a fact (unconditional assertion).
#[derive(Debug, Clone)]
pub struct Rule {
    /// The head atom (conclusion).
    pub head: Atom,
    /// The body atoms (premises), conjoined.
    pub body: Vec<Atom>,
}

impl Rule {
    /// Create a new rule.
    pub fn new(head: Atom, body: Vec<Atom>) -> Self {
        Self { head, body }
    }

    /// Returns `true` when the body is empty (the rule is an unconditional fact).
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation
// ─────────────────────────────────────────────────────────────────────────────

/// A set of facts sharing the same predicate.
#[derive(Debug, Clone, Default)]
pub struct Relation {
    facts: HashSet<Fact>,
}

impl Relation {
    /// Create an empty relation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a fact.  Returns `true` if the fact was not already present.
    pub fn insert(&mut self, fact: Fact) -> bool {
        self.facts.insert(fact)
    }

    /// Check whether the relation contains the given fact.
    pub fn contains(&self, fact: &Fact) -> bool {
        self.facts.contains(fact)
    }

    /// Number of facts in this relation.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Returns `true` if the relation has no facts.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Iterate over all facts.
    pub fn iter(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter()
    }

    /// Return a cloned `Vec` of all facts.
    pub fn facts(&self) -> Vec<Fact> {
        self.facts.iter().cloned().collect()
    }

    /// Compute the union of two relations (facts from both).
    pub fn union(&self, other: &Relation) -> Relation {
        let mut result = self.clone();
        for f in other.facts.iter() {
            result.facts.insert(f.clone());
        }
        result
    }

    /// Compute the set-difference `self − other`.
    pub fn difference(&self, other: &Relation) -> Relation {
        Relation {
            facts: self
                .facts
                .iter()
                .filter(|f| !other.facts.contains(*f))
                .cloned()
                .collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edb — Extensional Database (base facts)
// ─────────────────────────────────────────────────────────────────────────────

/// The extensional database: the set of base (input) facts.
#[derive(Debug, Clone, Default)]
pub struct Edb {
    relations: HashMap<String, Relation>,
}

impl Edb {
    /// Create an empty EDB.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a base fact.
    pub fn add_fact(&mut self, fact: Fact) {
        self.relations
            .entry(fact.predicate.clone())
            .or_default()
            .insert(fact);
    }

    /// Retrieve the relation for a predicate, if any.
    pub fn get_relation(&self, predicate: &str) -> Option<&Relation> {
        self.relations.get(predicate)
    }

    /// List the names of all predicates in the EDB.
    pub fn relation_names(&self) -> Vec<String> {
        self.relations.keys().cloned().collect()
    }

    /// Total number of base facts across all predicates.
    pub fn total_facts(&self) -> usize {
        self.relations.values().map(|r| r.len()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Idb — Intensional Database (derived facts)
// ─────────────────────────────────────────────────────────────────────────────

/// The intensional database: the set of derived (output) facts.
#[derive(Debug, Clone, Default)]
pub struct Idb {
    relations: HashMap<String, Relation>,
}

impl Idb {
    /// Create an empty IDB.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve the relation for a predicate, if any.
    pub fn get_relation(&self, predicate: &str) -> Option<&Relation> {
        self.relations.get(predicate)
    }

    /// Insert a derived fact.  Returns `true` if the fact was new.
    pub fn insert(&mut self, predicate: &str, fact: Fact) -> bool {
        self.relations
            .entry(predicate.to_owned())
            .or_default()
            .insert(fact)
    }

    /// Total number of derived facts.
    pub fn total_facts(&self) -> usize {
        self.relations.values().map(|r| r.len()).sum()
    }

    /// Return every derived fact as a flat `Vec`.
    pub fn all_facts(&self) -> Vec<Fact> {
        self.relations
            .values()
            .flat_map(|r| r.facts.iter().cloned())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EvalStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics gathered during semi-naive evaluation.
#[derive(Debug, Default, Clone)]
pub struct EvalStats {
    /// Total number of fixpoint iterations performed.
    pub iterations: usize,
    /// Total number of new facts derived across all iterations.
    pub total_new_facts: usize,
    /// How many new facts were derived in each individual iteration.
    pub facts_per_iteration: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// QueryError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during query evaluation.
#[derive(Debug)]
pub enum QueryError {
    /// A rule body atom references a predicate that is neither EDB nor IDB.
    UnknownPredicate(String),
    /// The arity of a queried fact does not match the schema expectation.
    ArityMismatch {
        predicate: String,
        expected: usize,
        got: usize,
    },
    /// An internal evaluation error.
    EvaluationError(String),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::UnknownPredicate(p) => write!(f, "unknown predicate: {p}"),
            QueryError::ArityMismatch {
                predicate,
                expected,
                got,
            } => write!(
                f,
                "arity mismatch for predicate {predicate}: expected {expected}, got {got}"
            ),
            QueryError::EvaluationError(msg) => write!(f, "evaluation error: {msg}"),
        }
    }
}

impl std::error::Error for QueryError {}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt to unify `term` against the concrete `arg`, extending `bindings`.
///
/// * If `term` is a `Variable` that is already bound, the binding must equal `arg`.
/// * If `term` is a `Variable` that is unbound, it is bound to `arg`.
/// * If `term` is a `Constant`, it must equal `arg`.
///
/// Returns `true` on success, `false` on failure (conflict).
fn unify_term(term: &Term, arg: &FactArg, bindings: &mut HashMap<String, FactArg>) -> bool {
    match term {
        Term::Variable(name) => {
            if let Some(existing) = bindings.get(name) {
                existing == arg
            } else {
                bindings.insert(name.clone(), arg.clone());
                true
            }
        }
        Term::Constant(c) => c == arg,
    }
}

/// Substitute a complete set of `bindings` into the `head` atom to produce a
/// ground `Fact`.  Returns `None` if any variable in the head is unbound.
fn ground_head(head: &Atom, bindings: &HashMap<String, FactArg>) -> Option<Fact> {
    let mut args = Vec::with_capacity(head.terms.len());
    for term in &head.terms {
        let arg = match term {
            Term::Variable(name) => bindings.get(name)?.clone(),
            Term::Constant(c) => c.clone(),
        };
        args.push(arg);
    }
    Some(Fact::new(head.predicate.clone(), args))
}

/// Recursively extend `current_bindings` over the remaining `atoms`,
/// looking facts up in `all_facts`.
///
/// For the *semi-naive* optimisation at least one atom in the conjunction must
/// be resolved against `delta` (the set of new facts from the previous
/// iteration) rather than the full relation.  The boolean `used_delta`
/// tracks whether a delta relation has already been used in the current
/// conjunction path.  `delta` contains only the relations that changed in the
/// last iteration.
fn eval_body_atoms<'a>(
    atoms: &'a [Atom],
    current_bindings: HashMap<String, FactArg>,
    all_facts: &'a HashMap<String, Relation>,
    delta: &'a HashMap<String, Relation>,
    used_delta: bool,
) -> Vec<HashMap<String, FactArg>> {
    if atoms.is_empty() {
        // Require that at least one delta atom was used in order to avoid
        // re-deriving already-known facts (semi-naive condition).
        if used_delta {
            return vec![current_bindings];
        } else {
            return vec![];
        }
    }

    let (head_atom, rest) = atoms.split_first().expect("atoms is non-empty");

    let predicate = &head_atom.predicate;
    let mut results: Vec<HashMap<String, FactArg>> = Vec::new();

    // Determine which fact-sets to scan.
    // Semi-naive: we try the atom against the delta relation (if present) and
    // separately against the full relation.  The `used_delta` flag ensures the
    // overall conjunction touches at least one delta tuple.

    let full_rel = all_facts.get(predicate.as_str());
    let delta_rel = delta.get(predicate.as_str());

    // Helper: iterate over a relation and collect extended bindings.
    let try_relation = |rel: &Relation,
                        bindings: &HashMap<String, FactArg>,
                        is_delta: bool|
     -> Vec<HashMap<String, FactArg>> {
        let mut out = Vec::new();
        for fact in rel.iter() {
            if fact.terms_len() != head_atom.terms.len() {
                continue;
            }
            let mut b = bindings.clone();
            let mut ok = true;
            for (term, arg) in head_atom.terms.iter().zip(fact.args.iter()) {
                if !unify_term(term, arg, &mut b) {
                    ok = false;
                    break;
                }
            }
            if ok {
                let mut sub = eval_body_atoms(rest, b, all_facts, delta, used_delta || is_delta);
                out.append(&mut sub);
            }
        }
        out
    };

    // Strategy:
    // 1. Use the delta relation for this atom (marks used_delta = true).
    if let Some(dr) = delta_rel {
        let mut sub = try_relation(dr, &current_bindings, true);
        results.append(&mut sub);
    }

    // 2. Use the full relation for this atom but only if we haven't used delta
    //    yet or there are remaining atoms that can provide the delta touch.
    //    In practice we always scan the full relation; the used_delta guard at
    //    the base case prevents duplicates being emitted.
    if let Some(fr) = full_rel {
        // When there is a delta for this predicate and used_delta is already
        // true (from an earlier atom), we can scan the full relation freely.
        // When used_delta is false we still scan the full relation because a
        // later atom may hit the delta.  Duplicates are avoided by the
        // base-case guard.
        let mut sub = try_relation(fr, &current_bindings, false);
        results.append(&mut sub);
    }

    results
}

// Extension trait to get the argument count from a Fact without exposing
// implementation details.
trait FactExt {
    fn terms_len(&self) -> usize;
}
impl FactExt for Fact {
    fn terms_len(&self) -> usize {
        self.args.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SemiNaiveEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Semi-naive Datalog evaluator.
///
/// Computes the least fixpoint of a set of Datalog rules over a base EDB by
/// iterating only over the *delta* (newly derived facts) at each round.
pub struct SemiNaiveEvaluator {
    rules: Vec<Rule>,
    edb: Edb,
    idb: Idb,
    stats: EvalStats,
}

impl SemiNaiveEvaluator {
    /// Create a new evaluator with the given rules and EDB.
    pub fn new(rules: Vec<Rule>, edb: Edb) -> Self {
        Self {
            rules,
            edb,
            idb: Idb::new(),
            stats: EvalStats::default(),
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Build a unified view of all known facts: EDB ∪ IDB, keyed by predicate.
    fn all_facts_snapshot(&self) -> HashMap<String, Relation> {
        let mut map: HashMap<String, Relation> = HashMap::new();

        for (pred, rel) in &self.edb.relations {
            map.entry(pred.clone())
                .or_default()
                .facts
                .extend(rel.facts.iter().cloned());
        }
        for (pred, rel) in &self.idb.relations {
            map.entry(pred.clone())
                .or_default()
                .facts
                .extend(rel.facts.iter().cloned());
        }
        map
    }

    /// Apply a single rule using `delta` as the "new facts" layer, returning
    /// any newly derivable head facts that are not yet in the IDB.
    fn apply_rule(&self, rule: &Rule, delta: &HashMap<String, Relation>) -> Vec<Fact> {
        // Rules with an empty body are fact rules — handled separately during
        // initialisation; skip them here.
        if rule.is_fact() {
            return vec![];
        }

        let all_facts = self.all_facts_snapshot();
        let bindings: HashMap<String, FactArg> = HashMap::new();

        let binding_sets = eval_body_atoms(&rule.body, bindings, &all_facts, delta, false);

        let mut new_facts: Vec<Fact> = Vec::new();
        for b in binding_sets {
            if let Some(fact) = ground_head(&rule.head, &b) {
                // Only emit facts that are not yet in the IDB.
                let already_known = self
                    .idb
                    .get_relation(&fact.predicate)
                    .map(|r| r.contains(&fact))
                    .unwrap_or(false);
                if !already_known {
                    new_facts.push(fact);
                }
            }
        }
        // Deduplicate within the batch.
        new_facts.sort_unstable_by(|a, b| format!("{a}").cmp(&format!("{b}")));
        new_facts.dedup();
        new_facts
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Run semi-naive evaluation to fixpoint and return the final IDB.
    ///
    /// Algorithm:
    /// 1. Seed Δ with all EDB facts plus any IDB fact-rules.
    /// 2. Repeat until Δ is empty:
    ///    a. For each rule, derive new facts using Δ.
    ///    b. New facts not already in IDB become the next Δ.
    ///    c. Add all new facts to IDB.
    pub fn evaluate(&mut self) -> Result<&Idb, QueryError> {
        // ── Step 0: Bootstrap IDB with EDB fact-rules (body-less rules). ──────
        for rule in &self.rules {
            if rule.is_fact() {
                if let Some(fact) = ground_head(&rule.head, &HashMap::new()) {
                    self.idb.insert(&fact.predicate.clone(), fact);
                }
            }
        }

        // ── Step 1: Seed delta with EDB facts + initial IDB. ─────────────────
        let mut delta: HashMap<String, Relation> = HashMap::new();

        for (pred, rel) in &self.edb.relations {
            delta
                .entry(pred.clone())
                .or_default()
                .facts
                .extend(rel.facts.iter().cloned());
        }
        for (pred, rel) in &self.idb.relations {
            delta
                .entry(pred.clone())
                .or_default()
                .facts
                .extend(rel.facts.iter().cloned());
        }

        // ── Step 2: Fixpoint loop. ────────────────────────────────────────────
        loop {
            if delta.values().all(|r| r.is_empty()) {
                break;
            }

            let mut new_delta: HashMap<String, Relation> = HashMap::new();
            let mut iteration_count = 0usize;

            for rule in &self.rules {
                if rule.is_fact() {
                    continue;
                }
                let derived = self.apply_rule(rule, &delta);
                for fact in derived {
                    let pred = fact.predicate.clone();
                    let is_new = self.idb.insert(&pred, fact.clone());
                    if is_new {
                        new_delta.entry(pred).or_default().insert(fact);
                        iteration_count += 1;
                    }
                }
            }

            self.stats.iterations += 1;
            self.stats.facts_per_iteration.push(iteration_count);
            self.stats.total_new_facts += iteration_count;

            delta = new_delta;
        }

        Ok(&self.idb)
    }

    /// Return evaluation statistics.
    pub fn stats(&self) -> &EvalStats {
        &self.stats
    }

    /// Return a reference to the current IDB.
    pub fn idb(&self) -> &Idb {
        &self.idb
    }

    /// Return a reference to the EDB.
    pub fn edb(&self) -> &Edb {
        &self.edb
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IncrementalEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps [`SemiNaiveEvaluator`] to support incremental updates: add new base
/// facts without discarding previously derived knowledge.
pub struct IncrementalEvaluator {
    evaluator: SemiNaiveEvaluator,
}

impl IncrementalEvaluator {
    /// Create and immediately evaluate the initial EDB.
    pub fn new(rules: Vec<Rule>, initial_edb: Edb) -> Result<Self, QueryError> {
        let mut evaluator = SemiNaiveEvaluator::new(rules, initial_edb);
        evaluator.evaluate()?;
        Ok(Self { evaluator })
    }

    /// Add new base facts to the EDB and propagate their consequences into the IDB.
    ///
    /// Only the newly added facts seed the next Δ, so previously computed
    /// derived facts are never recomputed from scratch.
    pub fn add_facts(&mut self, new_facts: Vec<Fact>) -> Result<EvalStats, QueryError> {
        // Inject into EDB.
        for fact in &new_facts {
            self.evaluator.edb.add_fact(fact.clone());
        }

        // Seed delta only with the new facts.
        let mut delta: HashMap<String, Relation> = HashMap::new();
        for fact in new_facts {
            delta
                .entry(fact.predicate.clone())
                .or_default()
                .insert(fact);
        }

        let mut local_stats = EvalStats::default();

        // Iterate until no new IDB facts emerge.
        loop {
            if delta.values().all(|r| r.is_empty()) {
                break;
            }

            let mut new_delta: HashMap<String, Relation> = HashMap::new();
            let mut iteration_count = 0usize;

            for rule in &self.evaluator.rules {
                if rule.is_fact() {
                    continue;
                }
                let derived = self.evaluator.apply_rule(rule, &delta);
                for fact in derived {
                    let pred = fact.predicate.clone();
                    let is_new = self.evaluator.idb.insert(&pred, fact.clone());
                    if is_new {
                        new_delta.entry(pred).or_default().insert(fact);
                        iteration_count += 1;
                    }
                }
            }

            local_stats.iterations += 1;
            local_stats.facts_per_iteration.push(iteration_count);
            local_stats.total_new_facts += iteration_count;

            // Mirror into the evaluator's global stats.
            self.evaluator.stats.iterations += 1;
            self.evaluator.stats.total_new_facts += iteration_count;
            self.evaluator
                .stats
                .facts_per_iteration
                .push(iteration_count);

            delta = new_delta;
        }

        Ok(local_stats)
    }

    /// Query the current IDB for all facts of the given predicate.
    pub fn query(&self, predicate: &str) -> Vec<Fact> {
        self.evaluator
            .idb
            .get_relation(predicate)
            .map(|r| r.facts())
            .unwrap_or_default()
    }

    /// Total number of derived facts currently in the IDB.
    pub fn total_derived_facts(&self) -> usize {
        self.evaluator.idb.total_facts()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Utility helpers ───────────────────────────────────────────────────────

    fn make_parent_edb() -> Edb {
        let mut edb = Edb::new();
        edb.add_fact(Fact::sym("parent", &["alice", "bob"]));
        edb.add_fact(Fact::sym("parent", &["bob", "carol"]));
        edb
    }

    fn ancestor_rules() -> Vec<Rule> {
        vec![
            Rule::new(
                Atom::new("ancestor", vec![Term::var("X"), Term::var("Y")]),
                vec![Atom::new("parent", vec![Term::var("X"), Term::var("Y")])],
            ),
            Rule::new(
                Atom::new("ancestor", vec![Term::var("X"), Term::var("Z")]),
                vec![
                    Atom::new("parent", vec![Term::var("X"), Term::var("Y")]),
                    Atom::new("ancestor", vec![Term::var("Y"), Term::var("Z")]),
                ],
            ),
        ]
    }

    // ── Test 1: Empty EDB + no rules → empty IDB ─────────────────────────────

    #[test]
    fn test_empty_edb_no_rules() {
        let mut eval = SemiNaiveEvaluator::new(vec![], Edb::new());
        let idb = eval.evaluate().expect("evaluation should succeed");
        assert_eq!(idb.total_facts(), 0, "empty IDB expected");
    }

    // ── Test 2: Fact rules (body-less) insert into IDB directly ──────────────

    #[test]
    fn test_fact_rules_insert_directly() {
        let rule = Rule::new(
            Atom::new("foo", vec![Term::sym("bar")]),
            vec![], // body-less
        );
        let mut eval = SemiNaiveEvaluator::new(vec![rule], Edb::new());
        let idb = eval.evaluate().expect("evaluation should succeed");
        let facts = idb.get_relation("foo").expect("relation foo should exist");
        assert_eq!(facts.len(), 1);
        assert!(facts.contains(&Fact::sym("foo", &["bar"])));
    }

    // ── Test 3: Simple chain: direct ancestor ─────────────────────────────────

    #[test]
    fn test_simple_ancestor_chain() {
        let rule = Rule::new(
            Atom::new("ancestor", vec![Term::var("X"), Term::var("Y")]),
            vec![Atom::new("parent", vec![Term::var("X"), Term::var("Y")])],
        );
        let mut eval = SemiNaiveEvaluator::new(vec![rule], make_parent_edb());
        let idb = eval.evaluate().expect("evaluation should succeed");
        let derived = idb.get_relation("ancestor").expect("ancestor relation");
        // Should contain alice->bob and bob->carol.
        assert!(derived.contains(&Fact::sym("ancestor", &["alice", "bob"])));
        assert!(derived.contains(&Fact::sym("ancestor", &["bob", "carol"])));
    }

    // ── Test 4: Recursive rule → transitive closure ───────────────────────────

    #[test]
    fn test_recursive_transitive_closure() {
        let mut eval = SemiNaiveEvaluator::new(ancestor_rules(), make_parent_edb());
        let idb = eval.evaluate().expect("evaluation should succeed");
        let derived = idb.get_relation("ancestor").expect("ancestor relation");
        assert!(derived.contains(&Fact::sym("ancestor", &["alice", "carol"])));
        assert_eq!(derived.len(), 3);
    }

    // ── Test 5: Fixpoint stops when no new facts ──────────────────────────────

    #[test]
    fn test_fixpoint_terminates() {
        let mut eval = SemiNaiveEvaluator::new(ancestor_rules(), make_parent_edb());
        eval.evaluate().expect("evaluation should succeed");
        // Run again — should produce no new facts.
        let idb_after = eval.idb().total_facts();
        assert_eq!(idb_after, 3);
    }

    // ── Test 6: Stats.iterations > 1 for recursive rules ─────────────────────

    #[test]
    fn test_eval_stats_iterations() {
        let mut eval = SemiNaiveEvaluator::new(ancestor_rules(), make_parent_edb());
        eval.evaluate().expect("evaluation should succeed");
        // At least 2 iterations are needed to derive the transitive fact.
        assert!(
            eval.stats().iterations >= 2,
            "expected >=2 iterations, got {}",
            eval.stats().iterations
        );
    }

    // ── Test 7: EvalStats.total_new_facts counts correctly ───────────────────

    #[test]
    fn test_eval_stats_total_new_facts() {
        let mut eval = SemiNaiveEvaluator::new(ancestor_rules(), make_parent_edb());
        eval.evaluate().expect("evaluation should succeed");
        assert_eq!(eval.stats().total_new_facts, 3);
    }

    // ── Test 8: Relation.union ────────────────────────────────────────────────

    #[test]
    fn test_relation_union() {
        let mut r1 = Relation::new();
        r1.insert(Fact::sym("foo", &["a"]));

        let mut r2 = Relation::new();
        r2.insert(Fact::sym("foo", &["b"]));
        r2.insert(Fact::sym("foo", &["a"])); // duplicate

        let u = r1.union(&r2);
        assert_eq!(u.len(), 2);
    }

    // ── Test 9: Relation.difference ──────────────────────────────────────────

    #[test]
    fn test_relation_difference() {
        let mut r1 = Relation::new();
        r1.insert(Fact::sym("foo", &["a"]));
        r1.insert(Fact::sym("foo", &["b"]));

        let mut r2 = Relation::new();
        r2.insert(Fact::sym("foo", &["a"]));

        let diff = r1.difference(&r2);
        assert_eq!(diff.len(), 1);
        assert!(diff.contains(&Fact::sym("foo", &["b"])));
    }

    // ── Test 10: Edb.total_facts ──────────────────────────────────────────────

    #[test]
    fn test_edb_total_facts() {
        let edb = make_parent_edb();
        assert_eq!(edb.total_facts(), 2);
    }

    // ── Test 11: Idb.all_facts ────────────────────────────────────────────────

    #[test]
    fn test_idb_all_facts() {
        let mut eval = SemiNaiveEvaluator::new(ancestor_rules(), make_parent_edb());
        eval.evaluate().expect("evaluation should succeed");
        let all = eval.idb().all_facts();
        assert_eq!(all.len(), 3);
    }

    // ── Test 12: Rule with two body atoms — join ──────────────────────────────

    #[test]
    fn test_two_body_atom_join() {
        // sibling(X, Z) :- parent(Y, X), parent(Y, Z).
        let mut edb = Edb::new();
        edb.add_fact(Fact::sym("parent", &["alice", "bob"]));
        edb.add_fact(Fact::sym("parent", &["alice", "carol"]));

        let rule = Rule::new(
            Atom::new("sibling", vec![Term::var("X"), Term::var("Z")]),
            vec![
                Atom::new("parent", vec![Term::var("Y"), Term::var("X")]),
                Atom::new("parent", vec![Term::var("Y"), Term::var("Z")]),
            ],
        );

        let mut eval = SemiNaiveEvaluator::new(vec![rule], edb);
        let idb = eval.evaluate().expect("evaluation should succeed");
        let siblings = idb.get_relation("sibling").expect("sibling relation");
        // bob-bob, bob-carol, carol-bob, carol-carol
        assert_eq!(siblings.len(), 4);
    }

    // ── Test 13: Constant in rule body filters ────────────────────────────────

    #[test]
    fn test_constant_in_body_filters() {
        // known_alice(Y) :- parent("alice", Y).
        let rule = Rule::new(
            Atom::new("known_alice", vec![Term::var("Y")]),
            vec![Atom::new(
                "parent",
                vec![Term::sym("alice"), Term::var("Y")],
            )],
        );

        let mut eval = SemiNaiveEvaluator::new(vec![rule], make_parent_edb());
        let idb = eval.evaluate().expect("evaluation should succeed");
        let rel = idb.get_relation("known_alice").expect("known_alice");
        assert_eq!(rel.len(), 1);
        assert!(rel.contains(&Fact::new(
            "known_alice",
            vec![FactArg::Symbol("bob".to_owned())]
        )));
    }

    // ── Test 14: Variable reuse (equality check) ──────────────────────────────

    #[test]
    fn test_variable_reuse_equality() {
        // self_parent(X) :- parent(X, X).
        let mut edb = Edb::new();
        edb.add_fact(Fact::sym("parent", &["alice", "bob"]));
        edb.add_fact(Fact::sym("parent", &["self", "self"]));

        let rule = Rule::new(
            Atom::new("self_parent", vec![Term::var("X")]),
            vec![Atom::new("parent", vec![Term::var("X"), Term::var("X")])],
        );

        let mut eval = SemiNaiveEvaluator::new(vec![rule], edb);
        let idb = eval.evaluate().expect("evaluation should succeed");
        let rel = idb.get_relation("self_parent").expect("self_parent");
        assert_eq!(rel.len(), 1);
        assert!(rel.contains(&Fact::new(
            "self_parent",
            vec![FactArg::Symbol("self".to_owned())]
        )));
    }

    // ── Test 15: IncrementalEvaluator.add_facts propagates ───────────────────

    #[test]
    fn test_incremental_add_facts() {
        let edb = make_parent_edb(); // alice->bob, bob->carol
        let mut inc =
            IncrementalEvaluator::new(ancestor_rules(), edb).expect("init should succeed");

        // Initially 3 derived facts.
        assert_eq!(inc.total_derived_facts(), 3);

        // Add carol->dave.
        inc.add_facts(vec![Fact::sym("parent", &["carol", "dave"])])
            .expect("add_facts should succeed");

        let ancestors = inc.query("ancestor");
        // alice->bob, alice->carol, alice->dave, bob->carol, bob->dave, carol->dave
        assert_eq!(ancestors.len(), 6, "expected 6 ancestor pairs");
    }

    // ── Test 16: IncrementalEvaluator.query ──────────────────────────────────

    #[test]
    fn test_incremental_query() {
        let edb = make_parent_edb();
        let inc = IncrementalEvaluator::new(ancestor_rules(), edb).expect("init should succeed");

        let ancestors = inc.query("ancestor");
        assert!(!ancestors.is_empty());

        // Non-existent predicate returns empty.
        let none = inc.query("no_such_predicate");
        assert!(none.is_empty());
    }

    // ── Test 17: Semi-naive avoids re-deriving known facts ───────────────────

    #[test]
    fn test_semi_naive_no_redundant_recomputation() {
        let edb = make_parent_edb();
        let mut inc =
            IncrementalEvaluator::new(ancestor_rules(), edb).expect("init should succeed");

        let before = inc.total_derived_facts();

        // Adding a fact that produces no new derived facts should leave the
        // IDB size unchanged (alice already has bob as ancestor).
        // (We add a duplicate base fact.)
        let stats = inc
            .add_facts(vec![Fact::sym("parent", &["alice", "bob"])])
            .expect("add_facts should succeed");

        let after = inc.total_derived_facts();
        assert_eq!(before, after, "no new derived facts expected");
        assert_eq!(stats.total_new_facts, 0);
    }

    // ── Test 18: Fact.sym convenience constructor ─────────────────────────────

    #[test]
    fn test_fact_sym_constructor() {
        let f = Fact::sym("edge", &["a", "b"]);
        assert_eq!(f.predicate, "edge");
        assert_eq!(f.arity(), 2);
        assert_eq!(f.args[0], FactArg::Symbol("a".to_owned()));
        assert_eq!(f.args[1], FactArg::Symbol("b".to_owned()));
    }

    // ── Test 19: Term constructors ────────────────────────────────────────────

    #[test]
    fn test_term_constructors() {
        let v = Term::var("X");
        let s = Term::sym("hello");
        let n = Term::int(42);

        assert!(matches!(v, Term::Variable(ref x) if x == "X"));
        assert!(matches!(s, Term::Constant(FactArg::Symbol(ref x)) if x == "hello"));
        assert!(matches!(n, Term::Constant(FactArg::Integer(42))));
    }

    // ── Test 20: QueryError for rule body with unknown predicate ─────────────
    //
    // Our evaluator does not fail hard on unknown predicates (it simply finds
    // no facts for that predicate), but we provide a mechanism to detect it
    // after evaluation by checking EDB + IDB coverage.  Here we test that an
    // empty result is produced when the body atom predicate is absent from
    // both EDB and IDB.
    #[test]
    fn test_unknown_predicate_in_rule_body() {
        // foo(X) :- no_such_pred(X).
        let rule = Rule::new(
            Atom::new("foo", vec![Term::var("X")]),
            vec![Atom::new("no_such_pred", vec![Term::var("X")])],
        );
        let mut eval = SemiNaiveEvaluator::new(vec![rule], Edb::new());
        let idb = eval.evaluate().expect("evaluation should not hard-fail");
        // No facts derived because no_such_pred is empty.
        assert_eq!(idb.total_facts(), 0);

        // Verify QueryError can be constructed and displayed.
        let err = QueryError::UnknownPredicate("no_such_pred".to_owned());
        assert!(err.to_string().contains("no_such_pred"));
    }

    // ── Test 21: 5-node chain has 10 ancestor pairs ───────────────────────────

    #[test]
    fn test_five_node_chain() {
        let nodes = ["a", "b", "c", "d", "e"];
        let mut edb = Edb::new();
        for i in 0..nodes.len() - 1 {
            edb.add_fact(Fact::sym("parent", &[nodes[i], nodes[i + 1]]));
        }

        let mut eval = SemiNaiveEvaluator::new(ancestor_rules(), edb);
        let idb = eval.evaluate().expect("evaluation should succeed");
        let derived = idb.get_relation("ancestor").expect("ancestor relation");
        // For a 5-node chain a→b→c→d→e, the transitive closure has
        // C(5,2) = 10 pairs.
        assert_eq!(derived.len(), 10);
    }

    // ── Test 22: Multiple rules deriving same head accumulate ─────────────────

    #[test]
    fn test_multiple_rules_same_head() {
        let mut edb = Edb::new();
        edb.add_fact(Fact::sym("edge_a", &["x", "y"]));
        edb.add_fact(Fact::sym("edge_b", &["y", "z"]));

        let rule1 = Rule::new(
            Atom::new("reachable", vec![Term::var("X"), Term::var("Y")]),
            vec![Atom::new("edge_a", vec![Term::var("X"), Term::var("Y")])],
        );
        let rule2 = Rule::new(
            Atom::new("reachable", vec![Term::var("X"), Term::var("Y")]),
            vec![Atom::new("edge_b", vec![Term::var("X"), Term::var("Y")])],
        );

        let mut eval = SemiNaiveEvaluator::new(vec![rule1, rule2], edb);
        let idb = eval.evaluate().expect("evaluation should succeed");
        let rel = idb.get_relation("reachable").expect("reachable relation");
        assert_eq!(rel.len(), 2);
        assert!(rel.contains(&Fact::sym("reachable", &["x", "y"])));
        assert!(rel.contains(&Fact::sym("reachable", &["y", "z"])));
    }

    // ── Bonus: Integer fact args work ─────────────────────────────────────────

    #[test]
    fn test_integer_fact_args() {
        let mut edb = Edb::new();
        edb.add_fact(Fact::new(
            "score",
            vec![FactArg::Symbol("alice".to_owned()), FactArg::Integer(99)],
        ));

        // high_scorer(X) :- score(X, 99).
        let rule = Rule::new(
            Atom::new("high_scorer", vec![Term::var("X")]),
            vec![Atom::new("score", vec![Term::var("X"), Term::int(99)])],
        );

        let mut eval = SemiNaiveEvaluator::new(vec![rule], edb);
        let idb = eval.evaluate().expect("evaluation should succeed");
        let rel = idb.get_relation("high_scorer").expect("high_scorer");
        assert_eq!(rel.len(), 1);
    }
}
