//! Arc-consistency constraint propagation (AC-3 algorithm) for logic variable domains.
//!
//! This module implements CSP (Constraint Satisfaction Problem) based inference
//! complementary to the probabilistic inference elsewhere in tensorlogic-infer.
//!
//! # Overview
//!
//! The AC-3 algorithm enforces arc-consistency across a constraint network:
//! - Each variable has a [`Domain`] of possible values.
//! - [`BinaryConstraint`]s relate pairs of variables.
//! - [`propagate_arc_consistency`] repeatedly prunes domains until a fixed-point.
//! - [`solve`] combines AC-3 with backtracking to enumerate solutions.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------

/// A finite domain of possible `f64` values for a logic variable.
///
/// Values are stored in insertion order. Equality comparisons use `f64::to_bits`
/// so that `NaN != NaN` and `-0.0 != 0.0` are maintained consistently with
/// bitwise identity. In practice, domains should not contain `NaN`.
#[derive(Debug, Clone)]
pub struct Domain {
    values: Vec<f64>,
}

impl Domain {
    /// Create a domain from an explicit list of values.
    ///
    /// Duplicate values (by bit-equality) are deduplicated while preserving
    /// insertion order.
    pub fn new(values: Vec<f64>) -> Self {
        let mut seen: HashSet<u64> = HashSet::new();
        let deduped: Vec<f64> = values
            .into_iter()
            .filter(|v| seen.insert(v.to_bits()))
            .collect();
        Self { values: deduped }
    }

    /// Create a domain from an inclusive range `[start, end]` with the given `step`.
    ///
    /// Values are generated as `start + k * step` for `k = 0, 1, …` while the
    /// generated value does not exceed `end`.  A `step` of zero or negative is
    /// silently treated as producing only `start` if `start <= end`.
    pub fn from_range(start: f64, end: f64, step: f64) -> Self {
        let mut values: Vec<f64> = Vec::new();
        if step <= 0.0 {
            if start <= end {
                values.push(start);
            }
            return Self { values };
        }
        let mut current = start;
        let epsilon = step * 1e-9;
        while current <= end + epsilon {
            values.push(current);
            current += step;
        }
        Self { values }
    }

    /// Convenience constructor: `{0.0, 1.0}`.
    pub fn boolean() -> Self {
        Self {
            values: vec![0.0, 1.0],
        }
    }

    /// Returns `true` if the domain contains no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the number of values in the domain.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns a slice of all values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Remove a specific value from the domain (bitwise equality).
    ///
    /// Returns `true` if the value was present and removed.
    pub fn remove_value(&mut self, val: f64) -> bool {
        let bits = val.to_bits();
        if let Some(pos) = self.values.iter().position(|v| v.to_bits() == bits) {
            self.values.remove(pos);
            true
        } else {
            false
        }
    }

    /// Retain only values for which `predicate` returns `true`.
    ///
    /// Returns the number of values that were removed.
    pub fn retain<F: Fn(f64) -> bool>(&mut self, predicate: F) -> usize {
        let before = self.values.len();
        self.values.retain(|&v| predicate(v));
        before - self.values.len()
    }

    /// Return the intersection of `self` and `other` as a new domain.
    pub fn intersect(&self, other: &Domain) -> Domain {
        let other_bits: HashSet<u64> = other.values.iter().map(|v| v.to_bits()).collect();
        let values: Vec<f64> = self
            .values
            .iter()
            .copied()
            .filter(|v| other_bits.contains(&v.to_bits()))
            .collect();
        Domain { values }
    }

    /// Return the union of `self` and `other` as a new domain (preserving order,
    /// with `self`'s values first, then any values from `other` not already present).
    pub fn union(&self, other: &Domain) -> Domain {
        let mut seen: HashSet<u64> = self.values.iter().map(|v| v.to_bits()).collect();
        let mut values = self.values.clone();
        for &v in &other.values {
            if seen.insert(v.to_bits()) {
                values.push(v);
            }
        }
        Domain { values }
    }
}

impl Default for Domain {
    fn default() -> Self {
        Self { values: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// ConstraintRelation
// ---------------------------------------------------------------------------

/// The relation enforced by a [`BinaryConstraint`] between variables `x` and `y`.
#[derive(Clone)]
pub enum ConstraintRelation {
    /// `x == y`
    Equal,
    /// `x != y`
    NotEqual,
    /// `x < y`
    LessThan,
    /// `x <= y`
    LessOrEqual,
    /// `x > y`
    GreaterThan,
    /// `x >= y`
    GreaterOrEqual,
    /// `|x - y| <= delta`
    Difference(f64),
    /// Arbitrary user-supplied relation.  Uses `Arc` so that `ConstraintRelation`
    /// can implement `Clone`.
    Custom(Arc<dyn Fn(f64, f64) -> bool + Send + Sync>),
}

impl std::fmt::Debug for ConstraintRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintRelation::Equal => write!(f, "Equal"),
            ConstraintRelation::NotEqual => write!(f, "NotEqual"),
            ConstraintRelation::LessThan => write!(f, "LessThan"),
            ConstraintRelation::LessOrEqual => write!(f, "LessOrEqual"),
            ConstraintRelation::GreaterThan => write!(f, "GreaterThan"),
            ConstraintRelation::GreaterOrEqual => write!(f, "GreaterOrEqual"),
            ConstraintRelation::Difference(d) => write!(f, "Difference({})", d),
            ConstraintRelation::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

impl ConstraintRelation {
    /// Evaluate whether `(x, y)` satisfies this relation.
    pub fn holds(&self, x: f64, y: f64) -> bool {
        match self {
            ConstraintRelation::Equal => (x - y).abs() < f64::EPSILON,
            ConstraintRelation::NotEqual => (x - y).abs() >= f64::EPSILON,
            ConstraintRelation::LessThan => x < y,
            ConstraintRelation::LessOrEqual => x <= y,
            ConstraintRelation::GreaterThan => x > y,
            ConstraintRelation::GreaterOrEqual => x >= y,
            ConstraintRelation::Difference(delta) => (x - y).abs() <= *delta,
            ConstraintRelation::Custom(f) => f(x, y),
        }
    }

    /// Return a new relation that is the *reverse* of this one, i.e., the
    /// relation that holds for `(y, x)` when this holds for `(x, y)`.
    ///
    /// Used by AC-3 to generate the arc `(Xj, Xi)` from a constraint stated
    /// as `(Xi, Xj)`.
    pub fn reversed(&self) -> ConstraintRelation {
        match self {
            ConstraintRelation::Equal => ConstraintRelation::Equal,
            ConstraintRelation::NotEqual => ConstraintRelation::NotEqual,
            ConstraintRelation::LessThan => ConstraintRelation::GreaterThan,
            ConstraintRelation::LessOrEqual => ConstraintRelation::GreaterOrEqual,
            ConstraintRelation::GreaterThan => ConstraintRelation::LessThan,
            ConstraintRelation::GreaterOrEqual => ConstraintRelation::LessOrEqual,
            ConstraintRelation::Difference(d) => ConstraintRelation::Difference(*d),
            ConstraintRelation::Custom(f) => {
                let f_clone = Arc::clone(f);
                ConstraintRelation::Custom(Arc::new(move |x, y| f_clone(y, x)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BinaryConstraint
// ---------------------------------------------------------------------------

/// A binary constraint between two named variables.
#[derive(Debug, Clone)]
pub struct BinaryConstraint {
    /// The "left-hand" variable name.
    pub var_x: String,
    /// The "right-hand" variable name.
    pub var_y: String,
    /// The relation that must hold between the two variables' values.
    pub relation: ConstraintRelation,
}

impl BinaryConstraint {
    /// Construct a new `BinaryConstraint`.
    pub fn new(
        var_x: impl Into<String>,
        var_y: impl Into<String>,
        relation: ConstraintRelation,
    ) -> Self {
        Self {
            var_x: var_x.into(),
            var_y: var_y.into(),
            relation,
        }
    }
}

// ---------------------------------------------------------------------------
// ConstraintNetwork
// ---------------------------------------------------------------------------

/// A constraint network (CSP): variables with domains and binary constraints.
pub struct ConstraintNetwork {
    variables: HashMap<String, Domain>,
    constraints: Vec<BinaryConstraint>,
}

impl ConstraintNetwork {
    /// Create an empty constraint network.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Register a variable with its initial domain.
    ///
    /// If the variable already exists its domain is replaced.
    pub fn add_variable(&mut self, name: impl Into<String>, domain: Domain) {
        self.variables.insert(name.into(), domain);
    }

    /// Add a binary constraint to the network.
    ///
    /// Both `var_x` and `var_y` should have been registered via
    /// `add_variable` before running propagation.
    pub fn add_constraint(&mut self, constraint: BinaryConstraint) {
        self.constraints.push(constraint);
    }

    /// Number of variables registered.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Number of constraints in the network.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Borrow the domain of a variable, or `None` if not registered.
    pub fn domain(&self, var: &str) -> Option<&Domain> {
        self.variables.get(var)
    }

    /// Returns an iterator over all variable names.
    pub fn variable_names(&self) -> impl Iterator<Item = &str> {
        self.variables.keys().map(|s| s.as_str())
    }
}

impl Default for ConstraintNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PropagationResult
// ---------------------------------------------------------------------------

/// Outcome of running [`propagate_arc_consistency`].
#[derive(Debug)]
pub struct PropagationResult {
    /// `false` if any domain became empty (network is inconsistent).
    pub consistent: bool,
    /// Number of times an arc was dequeued and processed.
    pub iterations: usize,
    /// Total number of domain values pruned across all variables.
    pub pruned: usize,
    /// Names of variables whose domain became empty.
    pub empty_domains: Vec<String>,
}

// ---------------------------------------------------------------------------
// SolveStats
// ---------------------------------------------------------------------------

/// Statistics from a full backtracking solve.
#[derive(Debug)]
pub struct SolveStats {
    /// The result from the final propagation pass.
    pub propagation_result: PropagationResult,
    /// Number of complete solutions found.
    pub solutions_found: usize,
    /// Number of times the search backtracked.
    pub backtrack_count: usize,
    /// Number of search-tree nodes explored (assignments attempted).
    pub nodes_explored: usize,
}

// ---------------------------------------------------------------------------
// CspConfig / VarOrdering
// ---------------------------------------------------------------------------

/// Variable ordering heuristic for the backtracking solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarOrdering {
    /// Select variables in lexicographic order of their names.
    Lexicographic,
    /// Minimum Remaining Values: select the variable with the smallest domain.
    MinRemainingValues,
    /// Degree Heuristic: select the variable involved in the most constraints.
    DegreeHeuristic,
}

/// Configuration for the backtracking CSP solver.
pub struct CspConfig {
    /// Maximum number of solutions to return.  `0` means find all solutions.
    pub max_solutions: usize,
    /// If `true`, run AC-3 arc consistency at the root before search.
    pub use_arc_consistency: bool,
    /// If `true`, apply forward checking (domain filtering) after each
    /// variable assignment.
    pub forward_checking: bool,
    /// Variable selection ordering strategy.
    pub variable_ordering: VarOrdering,
}

impl Default for CspConfig {
    fn default() -> Self {
        Self {
            max_solutions: 0,
            use_arc_consistency: true,
            forward_checking: true,
            variable_ordering: VarOrdering::MinRemainingValues,
        }
    }
}

// ---------------------------------------------------------------------------
// AC-3 implementation
// ---------------------------------------------------------------------------

/// An arc in the AC-3 work-queue: `(Xi_name, Xj_name, relation_for_Xi_given_Xj)`.
///
/// The relation stored is the "one-sided" version: given a value `a` in
/// `D(Xi)`, we need to find some `b` in `D(Xj)` such that `relation.holds(a, b)`
/// is `true`.
#[derive(Debug)]
struct ArcEntry {
    xi: String,
    xj: String,
    /// The relation to check: `∃ b ∈ D(Xj) : relation.holds(a, b)`.
    relation: ConstraintRelation,
}

/// Run AC-3 arc-consistency on the network, modifying domains in-place.
///
/// Returns a [`PropagationResult`] describing what changed and whether the
/// network is still consistent.
pub fn propagate_arc_consistency(network: &mut ConstraintNetwork) -> PropagationResult {
    // Build the initial queue: for every binary constraint, enqueue both
    // directions (Xi → Xj) and (Xj → Xi).
    let mut queue: VecDeque<ArcEntry> = VecDeque::new();

    for constraint in &network.constraints {
        // Forward arc: Xi must have a support in Xj.
        queue.push_back(ArcEntry {
            xi: constraint.var_x.clone(),
            xj: constraint.var_y.clone(),
            relation: constraint.relation.clone(),
        });
        // Backward arc: Xj must have a support in Xi (reversed relation).
        queue.push_back(ArcEntry {
            xi: constraint.var_y.clone(),
            xj: constraint.var_x.clone(),
            relation: constraint.relation.reversed(),
        });
    }

    let mut iterations: usize = 0;
    let mut pruned: usize = 0;
    let mut empty_domains: Vec<String> = Vec::new();

    while let Some(arc) = queue.pop_front() {
        iterations += 1;

        // Borrow both domains.  If either variable is unknown, skip.
        let xi_values: Vec<f64> = match network.variables.get(&arc.xi) {
            Some(d) => d.values().to_vec(),
            None => continue,
        };
        let xj_values: Vec<f64> = match network.variables.get(&arc.xj) {
            Some(d) => d.values().to_vec(),
            None => continue,
        };

        // Identify values in D(Xi) that have no support in D(Xj).
        let to_remove: Vec<f64> = xi_values
            .iter()
            .copied()
            .filter(|&a| !xj_values.iter().any(|&b| arc.relation.holds(a, b)))
            .collect();

        if to_remove.is_empty() {
            // No pruning: arc is already consistent.
            continue;
        }

        // Apply pruning.
        let xi_domain = network
            .variables
            .get_mut(&arc.xi)
            .expect("variable must exist after earlier borrow");
        for val in &to_remove {
            xi_domain.remove_value(*val);
        }
        pruned += to_remove.len();

        // Check for empty domain (inconsistency).
        if xi_domain.is_empty() {
            if !empty_domains.contains(&arc.xi) {
                empty_domains.push(arc.xi.clone());
            }
            // Continue processing remaining arcs to find ALL empty domains.
            continue;
        }

        // Re-enqueue all arcs (Xk, Xi) for every constraint involving Xi,
        // except the arc in the other direction (Xj, Xi) that we just came from.
        let xi_name = arc.xi.clone();
        let xj_name = arc.xj.clone();

        // Collect the new arcs to enqueue to avoid borrowing issues.
        let new_arcs: Vec<ArcEntry> = network
            .constraints
            .iter()
            .flat_map(|c| {
                let mut arcs = Vec::new();
                // c involves Xi as var_x → enqueue (var_y → var_x).
                if c.var_x == xi_name && c.var_y != xj_name {
                    arcs.push(ArcEntry {
                        xi: c.var_y.clone(),
                        xj: xi_name.clone(),
                        relation: c.relation.reversed(),
                    });
                }
                // c involves Xi as var_y → enqueue (var_x → var_y).
                if c.var_y == xi_name && c.var_x != xj_name {
                    arcs.push(ArcEntry {
                        xi: c.var_x.clone(),
                        xj: xi_name.clone(),
                        relation: c.relation.clone(),
                    });
                }
                arcs
            })
            .collect();

        for new_arc in new_arcs {
            queue.push_back(new_arc);
        }
    }

    let consistent = empty_domains.is_empty();
    PropagationResult {
        consistent,
        iterations,
        pruned,
        empty_domains,
    }
}

// ---------------------------------------------------------------------------
// Backtracking solver helpers
// ---------------------------------------------------------------------------

/// Choose the next unassigned variable according to the configured ordering.
fn select_variable(
    unassigned: &[String],
    domains: &HashMap<String, Domain>,
    constraints: &[BinaryConstraint],
    ordering: &VarOrdering,
) -> String {
    match ordering {
        VarOrdering::Lexicographic => {
            let mut sorted = unassigned.to_vec();
            sorted.sort();
            sorted
                .into_iter()
                .next()
                .expect("select_variable called with non-empty unassigned list")
        }
        VarOrdering::MinRemainingValues => unassigned
            .iter()
            .min_by_key(|v| domains.get(*v).map(|d| d.len()).unwrap_or(usize::MAX))
            .cloned()
            .expect("select_variable called with non-empty unassigned list"),
        VarOrdering::DegreeHeuristic => {
            // Count constraints involving each unassigned variable.
            let unassigned_set: HashSet<&String> = unassigned.iter().collect();
            unassigned
                .iter()
                .max_by_key(|v| {
                    constraints
                        .iter()
                        .filter(|c| {
                            (&c.var_x == *v && unassigned_set.contains(&c.var_y))
                                || (&c.var_y == *v && unassigned_set.contains(&c.var_x))
                        })
                        .count()
                })
                .cloned()
                .expect("select_variable called with non-empty unassigned list")
        }
    }
}

/// Apply forward checking after assigning `var = value`.
///
/// Returns `false` if any neighbour domain becomes empty (triggering backtrack).
fn forward_check(
    var: &str,
    value: f64,
    domains: &mut HashMap<String, Domain>,
    constraints: &[BinaryConstraint],
    assigned: &HashMap<String, f64>,
) -> bool {
    for constraint in constraints {
        // Determine the "other" variable w.r.t. the just-assigned `var`.
        let (other_var, rel) = if constraint.var_x == var {
            (constraint.var_y.as_str(), constraint.relation.clone())
        } else if constraint.var_y == var {
            (constraint.var_x.as_str(), constraint.relation.reversed())
        } else {
            continue;
        };

        // Only prune domains of unassigned variables.
        if assigned.contains_key(other_var) {
            continue;
        }

        let other_domain = match domains.get_mut(other_var) {
            Some(d) => d,
            None => continue,
        };

        // Remove values from other_domain that are not consistent with `value`.
        // "rel" is from the perspective of `var` → `other_var`, i.e.,
        // rel.holds(value, b) must be true for b to be kept in other_domain.
        other_domain.retain(|b| rel.holds(value, b));

        if other_domain.is_empty() {
            return false;
        }
    }
    true
}

/// Recursive backtracking search.
///
/// `domains` holds the *current* (possibly pruned) domains.
/// `assigned` holds the current partial assignment.
/// `solutions` accumulates complete solutions.
fn backtrack(
    unassigned: &mut Vec<String>,
    domains: &mut HashMap<String, Domain>,
    assigned: &mut HashMap<String, f64>,
    constraints: &[BinaryConstraint],
    config: &CspConfig,
    solutions: &mut Vec<HashMap<String, f64>>,
    stats: &mut SolveStats,
) {
    // Check solution limit.
    if config.max_solutions != 0 && solutions.len() >= config.max_solutions {
        return;
    }

    // Base case: all variables assigned → record solution.
    if unassigned.is_empty() {
        solutions.push(assigned.clone());
        stats.solutions_found += 1;
        return;
    }

    // Select next variable.
    let var = select_variable(unassigned, domains, constraints, &config.variable_ordering);

    // Remove from unassigned list.
    unassigned.retain(|v| v != &var);

    // Snapshot the domain for restoration on backtrack.
    let original_domain = domains.get(&var).cloned().unwrap_or_default();

    // Try each value in the variable's current domain.
    let candidates: Vec<f64> = domains
        .get(&var)
        .map(|d| d.values().to_vec())
        .unwrap_or_default();

    for value in candidates {
        // Check solution limit before exploring further.
        if config.max_solutions != 0 && solutions.len() >= config.max_solutions {
            break;
        }

        stats.nodes_explored += 1;

        // Assign.
        assigned.insert(var.clone(), value);

        // Snapshot all other domains for forward-checking rollback.
        let domain_snapshot: HashMap<String, Domain> = if config.forward_checking {
            domains.clone()
        } else {
            HashMap::new()
        };

        // Forward check.
        let fc_ok = if config.forward_checking {
            forward_check(&var, value, domains, constraints, assigned)
        } else {
            // Without forward checking: still verify consistency of the
            // current assignment against constraints involving only assigned vars.
            constraints.iter().all(|c| {
                let x_assigned = assigned.get(&c.var_x);
                let y_assigned = assigned.get(&c.var_y);
                match (x_assigned, y_assigned) {
                    (Some(&xv), Some(&yv)) => c.relation.holds(xv, yv),
                    _ => true, // incomplete assignment: skip for now
                }
            })
        };

        if fc_ok {
            backtrack(
                unassigned,
                domains,
                assigned,
                constraints,
                config,
                solutions,
                stats,
            );
        } else {
            stats.backtrack_count += 1;
        }

        // Undo assignment.
        assigned.remove(&var);

        // Restore domains if we were doing forward checking.
        if config.forward_checking {
            for (k, v) in domain_snapshot {
                domains.insert(k, v);
            }
        }
    }

    // Restore unassigned list.
    unassigned.push(var.clone());
    // Re-sort isn't needed for correctness; ordering is re-applied each call.

    // Restore domain of current variable.
    domains.insert(var, original_domain);
}

// ---------------------------------------------------------------------------
// Public solver
// ---------------------------------------------------------------------------

/// Solve the CSP using backtracking search (optionally with AC-3).
///
/// Returns a list of complete assignments (each mapping variable name → value)
/// and [`SolveStats`] describing the search effort.
pub fn solve(
    network: &ConstraintNetwork,
    config: &CspConfig,
) -> (Vec<HashMap<String, f64>>, SolveStats) {
    // Clone domains so we can modify them without affecting the original network.
    let mut domains: HashMap<String, Domain> = network.variables.clone();
    let constraints = network.constraints.clone();

    // Optionally run AC-3 to prune domains before search.
    let propagation_result = if config.use_arc_consistency {
        // Build a temporary mutable network for propagation.
        let mut temp_network = ConstraintNetwork {
            variables: domains.clone(),
            constraints: constraints.clone(),
        };
        let result = propagate_arc_consistency(&mut temp_network);
        // Update domains with pruned versions.
        domains = temp_network.variables;
        result
    } else {
        PropagationResult {
            consistent: true,
            iterations: 0,
            pruned: 0,
            empty_domains: Vec::new(),
        }
    };

    let mut stats = SolveStats {
        propagation_result,
        solutions_found: 0,
        backtrack_count: 0,
        nodes_explored: 0,
    };

    // If already inconsistent, return early.
    if !stats.propagation_result.consistent {
        return (Vec::new(), stats);
    }

    // Build the initial unassigned variable list.
    let mut unassigned: Vec<String> = domains.keys().cloned().collect();
    let mut assigned: HashMap<String, f64> = HashMap::new();
    let mut solutions: Vec<HashMap<String, f64>> = Vec::new();

    backtrack(
        &mut unassigned,
        &mut domains,
        &mut assigned,
        &constraints,
        config,
        &mut solutions,
        &mut stats,
    );

    (solutions, stats)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a two-variable network.
    fn two_var_network(
        domain_x: Domain,
        domain_y: Domain,
        relation: ConstraintRelation,
    ) -> ConstraintNetwork {
        let mut net = ConstraintNetwork::new();
        net.add_variable("x", domain_x);
        net.add_variable("y", domain_y);
        net.add_constraint(BinaryConstraint::new("x", "y", relation));
        net
    }

    // -----------------------------------------------------------------------
    // Domain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_domain_boolean_has_two_values() {
        let d = Domain::boolean();
        assert_eq!(d.len(), 2);
        let vals = d.values();
        assert!((vals[0] - 0.0).abs() < f64::EPSILON);
        assert!((vals[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_domain_from_range_basic() {
        let d = Domain::from_range(0.0, 4.0, 1.0);
        assert_eq!(d.len(), 5); // 0, 1, 2, 3, 4
        assert!((d.values()[0] - 0.0).abs() < 1e-9);
        assert!((d.values()[4] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_domain_from_range_fractional_step() {
        let d = Domain::from_range(0.0, 1.0, 0.5);
        assert_eq!(d.len(), 3); // 0.0, 0.5, 1.0
    }

    #[test]
    fn test_domain_retain_removes_correct_values() {
        let mut d = Domain::new(vec![1.0, 2.0, 3.0, 4.0]);
        let removed = d.retain(|v| v < 3.0);
        assert_eq!(removed, 2);
        assert_eq!(d.len(), 2);
        assert!(d.values().iter().all(|&v| v < 3.0));
    }

    #[test]
    fn test_domain_remove_value() {
        let mut d = Domain::new(vec![1.0, 2.0, 3.0]);
        assert!(d.remove_value(2.0));
        assert!(!d.remove_value(99.0));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_domain_intersect() {
        let a = Domain::new(vec![1.0, 2.0, 3.0]);
        let b = Domain::new(vec![2.0, 3.0, 4.0]);
        let c = a.intersect(&b);
        assert_eq!(c.len(), 2);
        assert!(c.values().contains(&2.0));
        assert!(c.values().contains(&3.0));
    }

    #[test]
    fn test_domain_union() {
        let a = Domain::new(vec![1.0, 2.0]);
        let b = Domain::new(vec![2.0, 3.0]);
        let c = a.union(&b);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn test_domain_deduplication_on_new() {
        let d = Domain::new(vec![1.0, 1.0, 2.0]);
        assert_eq!(d.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Propagation: trivial cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_network_is_consistent() {
        let mut net = ConstraintNetwork::new();
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        assert_eq!(result.pruned, 0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_single_variable_no_constraints_consistent() {
        let mut net = ConstraintNetwork::new();
        net.add_variable("x", Domain::from_range(0.0, 5.0, 1.0));
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        assert_eq!(result.pruned, 0);
    }

    // -----------------------------------------------------------------------
    // Propagation: equality
    // -----------------------------------------------------------------------

    #[test]
    fn test_equal_constraint_prunes_to_intersection() {
        let mut net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0]),
            Domain::new(vec![2.0, 3.0, 4.0]),
            ConstraintRelation::Equal,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        // After AC-3: x ∈ {2,3}, y ∈ {2,3}
        let dx = net.domain("x").expect("x must exist");
        let dy = net.domain("y").expect("y must exist");
        assert_eq!(dx.len(), 2);
        assert_eq!(dy.len(), 2);
        assert!(result.pruned > 0);
    }

    // -----------------------------------------------------------------------
    // Propagation: not-equal
    // -----------------------------------------------------------------------

    #[test]
    fn test_not_equal_boolean_domain() {
        // x != y, both in {0, 1}: AC-3 cannot prune anything because each
        // value in D(x) has a support in D(y) (0 supported by 1, 1 by 0).
        let mut net = two_var_network(
            Domain::boolean(),
            Domain::boolean(),
            ConstraintRelation::NotEqual,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        // Domains should remain full.
        assert_eq!(net.domain("x").expect("x").len(), 2);
        assert_eq!(net.domain("y").expect("y").len(), 2);
    }

    #[test]
    fn test_not_equal_single_value_forces_removal() {
        // x != y, D(x)={1}, D(y)={1,2}: AC-3 should leave D(y)={2}.
        let mut net = two_var_network(
            Domain::new(vec![1.0]),
            Domain::new(vec![1.0, 2.0]),
            ConstraintRelation::NotEqual,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        let dy = net.domain("y").expect("y");
        assert_eq!(dy.len(), 1);
        assert!((dy.values()[0] - 2.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Propagation: less-than
    // -----------------------------------------------------------------------

    #[test]
    fn test_less_than_prunes_domains() {
        // x < y, D(x)={1,2,3}, D(y)={1,2,3}
        // After AC-3: x cannot be 3 (no y > 3 exists), y cannot be 1 (no x < 1 exists).
        let mut net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0]),
            Domain::new(vec![1.0, 2.0, 3.0]),
            ConstraintRelation::LessThan,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        let dx = net.domain("x").expect("x");
        let dy = net.domain("y").expect("y");
        // x=3 has no support (needs y>3, but max is 3).
        assert!(!dx.values().contains(&3.0));
        // y=1 has no support (needs x<1, but min is 1).
        assert!(!dy.values().contains(&1.0));
    }

    // -----------------------------------------------------------------------
    // Propagation: less-or-equal
    // -----------------------------------------------------------------------

    #[test]
    fn test_less_or_equal_overlapping_domains() {
        // x <= y, D(x)={1,2,3}, D(y)={2,3,4}
        let mut net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0]),
            Domain::new(vec![2.0, 3.0, 4.0]),
            ConstraintRelation::LessOrEqual,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        // x can still be 1,2,3 (all have a y >= themselves).
        let dx = net.domain("x").expect("x");
        assert_eq!(dx.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Propagation: chain
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_propagation_x_lt_y_lt_z() {
        // x < y < z, all in {1,2,3}
        // AC-3 should propagate: z=1 invalid (no y<1), x=3 invalid (no y>3),
        // then y=1 invalid (no x<1 after x pruning), y=3 invalid (no z>3).
        let mut net = ConstraintNetwork::new();
        net.add_variable("x", Domain::new(vec![1.0, 2.0, 3.0]));
        net.add_variable("y", Domain::new(vec![1.0, 2.0, 3.0]));
        net.add_variable("z", Domain::new(vec![1.0, 2.0, 3.0]));
        net.add_constraint(BinaryConstraint::new(
            "x",
            "y",
            ConstraintRelation::LessThan,
        ));
        net.add_constraint(BinaryConstraint::new(
            "y",
            "z",
            ConstraintRelation::LessThan,
        ));

        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        // There exists a solution: x=1, y=2, z=3 — so all domains non-empty.
        assert!(!net.domain("x").expect("x").is_empty());
        assert!(!net.domain("y").expect("y").is_empty());
        assert!(!net.domain("z").expect("z").is_empty());
        // After full propagation x can only be 1 (only x < y possible with {1,2,3}→{2,3}→{3}).
        let dx = net.domain("x").expect("x");
        assert!(!dx.values().contains(&3.0));
    }

    // -----------------------------------------------------------------------
    // Propagation: inconsistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_inconsistency_detected_empty_domain() {
        // x < y, but D(x)={3}, D(y)={1,2}: no y > 3 → inconsistent.
        let mut net = two_var_network(
            Domain::new(vec![3.0]),
            Domain::new(vec![1.0, 2.0]),
            ConstraintRelation::LessThan,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(!result.consistent);
        assert!(!result.empty_domains.is_empty());
    }

    // -----------------------------------------------------------------------
    // Propagation: pruned count
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagation_result_pruned_counts_correctly() {
        // x == y, D(x)={1,2,3,4}, D(y)={3,4,5}
        // Expected: x prunes {1,2} (no support), y prunes {5} (no support) → 3 pruned.
        let mut net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0, 4.0]),
            Domain::new(vec![3.0, 4.0, 5.0]),
            ConstraintRelation::Equal,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        assert_eq!(result.pruned, 3);
    }

    // -----------------------------------------------------------------------
    // Difference constraint
    // -----------------------------------------------------------------------

    #[test]
    fn test_difference_constraint_within_tolerance() {
        // |x - y| <= 0.5, D(x)={1.0, 2.0, 3.0}, D(y)={1.0, 2.0, 3.0}
        // Every value has a support (itself), so no pruning.
        let mut net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0]),
            Domain::new(vec![1.0, 2.0, 3.0]),
            ConstraintRelation::Difference(0.5),
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        assert_eq!(result.pruned, 0);
    }

    #[test]
    fn test_difference_constraint_prunes_far_values() {
        // |x - y| <= 0.5, D(x)={1.0}, D(y)={1.0, 5.0}
        // y=5 has no support (|1-5|=4 > 0.5).
        let mut net = two_var_network(
            Domain::new(vec![1.0]),
            Domain::new(vec![1.0, 5.0]),
            ConstraintRelation::Difference(0.5),
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        assert_eq!(net.domain("y").expect("y").len(), 1);
    }

    // -----------------------------------------------------------------------
    // Custom constraint
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_constraint_function() {
        // Custom: x + y == 3.0
        let rel = ConstraintRelation::Custom(Arc::new(|x, y| (x + y - 3.0).abs() < 1e-9));
        let mut net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0]),
            Domain::new(vec![1.0, 2.0, 3.0]),
            rel,
        );
        let result = propagate_arc_consistency(&mut net);
        assert!(result.consistent);
        // x=3 needs y=0 (not in domain) → pruned; y=3 needs x=0 → pruned.
        let dx = net.domain("x").expect("x");
        let dy = net.domain("y").expect("y");
        assert!(!dx.values().contains(&3.0));
        assert!(!dy.values().contains(&3.0));
    }

    // -----------------------------------------------------------------------
    // Solver: single solution
    // -----------------------------------------------------------------------

    #[test]
    fn test_solver_finds_single_solution_equality() {
        // x == y, D(x)={42.0}, D(y)={42.0}
        let net = two_var_network(
            Domain::new(vec![42.0]),
            Domain::new(vec![42.0]),
            ConstraintRelation::Equal,
        );
        let config = CspConfig::default();
        let (solutions, stats) = solve(&net, &config);
        assert_eq!(solutions.len(), 1);
        assert_eq!(stats.solutions_found, 1);
        let sol = &solutions[0];
        assert!((sol["x"] - 42.0).abs() < f64::EPSILON);
        assert!((sol["y"] - 42.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Solver: all solutions for x != y on boolean domain
    // -----------------------------------------------------------------------

    #[test]
    fn test_solver_finds_all_solutions_not_equal_boolean() {
        // x != y, D(x)={0,1}, D(y)={0,1} → solutions: (0,1) and (1,0)
        let net = two_var_network(
            Domain::boolean(),
            Domain::boolean(),
            ConstraintRelation::NotEqual,
        );
        let config = CspConfig {
            max_solutions: 0,
            ..CspConfig::default()
        };
        let (solutions, stats) = solve(&net, &config);
        assert_eq!(solutions.len(), 2, "expected exactly 2 solutions");
        assert_eq!(stats.solutions_found, 2);
    }

    // -----------------------------------------------------------------------
    // Solver: max_solutions limit
    // -----------------------------------------------------------------------

    #[test]
    fn test_solver_max_solutions_stops_early() {
        // x != y, D(x)={0,1}, D(y)={0,1}: two solutions, but limit to 1.
        let net = two_var_network(
            Domain::boolean(),
            Domain::boolean(),
            ConstraintRelation::NotEqual,
        );
        let config = CspConfig {
            max_solutions: 1,
            ..CspConfig::default()
        };
        let (solutions, _stats) = solve(&net, &config);
        assert_eq!(solutions.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Solver: MRV ordering selects smallest domain first
    // -----------------------------------------------------------------------

    #[test]
    fn test_mrv_ordering_selects_smallest_domain() {
        // Build two variables, x has 1 value, y has 3; MRV should pick x first.
        let domains: HashMap<String, Domain> = [
            ("x".to_string(), Domain::new(vec![1.0])),
            ("y".to_string(), Domain::new(vec![1.0, 2.0, 3.0])),
        ]
        .into_iter()
        .collect();

        let constraints: Vec<BinaryConstraint> = Vec::new();
        let unassigned = vec!["x".to_string(), "y".to_string()];
        let chosen = select_variable(
            &unassigned,
            &domains,
            &constraints,
            &VarOrdering::MinRemainingValues,
        );
        assert_eq!(chosen, "x");
    }

    // -----------------------------------------------------------------------
    // Solver: degree heuristic
    // -----------------------------------------------------------------------

    #[test]
    fn test_degree_heuristic_selects_most_constrained() {
        // Three variables: x is in 2 constraints, y in 1, z in 1.
        // Degree heuristic should pick x.
        let domains: HashMap<String, Domain> = [
            ("x".to_string(), Domain::new(vec![1.0, 2.0])),
            ("y".to_string(), Domain::new(vec![1.0, 2.0])),
            ("z".to_string(), Domain::new(vec![1.0, 2.0])),
        ]
        .into_iter()
        .collect();

        let constraints = vec![
            BinaryConstraint::new("x", "y", ConstraintRelation::NotEqual),
            BinaryConstraint::new("x", "z", ConstraintRelation::NotEqual),
        ];

        let unassigned = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let chosen = select_variable(
            &unassigned,
            &domains,
            &constraints,
            &VarOrdering::DegreeHeuristic,
        );
        assert_eq!(chosen, "x");
    }

    // -----------------------------------------------------------------------
    // Solver: backtrack_count > 0 for a conflicting search
    // -----------------------------------------------------------------------

    #[test]
    fn test_solver_backtrack_count_nonzero_for_conflicted_search() {
        // x + y == 3 (custom), D(x)={1,2,3}, D(y)={1,2,3}
        // The solver must backtrack when it tries x=1,y=1 (sum=2≠3), etc.
        let rel = ConstraintRelation::Custom(Arc::new(|x, y| (x + y - 3.0).abs() < 1e-9));
        let net = two_var_network(
            Domain::new(vec![1.0, 2.0, 3.0]),
            Domain::new(vec![1.0, 2.0, 3.0]),
            rel,
        );
        let config = CspConfig {
            use_arc_consistency: false,
            forward_checking: false,
            max_solutions: 0,
            variable_ordering: VarOrdering::Lexicographic,
        };
        let (solutions, stats) = solve(&net, &config);
        // Solutions: (1,2), (2,1) — x and y from {1,2,3}, sum must be 3.
        assert!(!solutions.is_empty());
        assert!(stats.backtrack_count > 0 || stats.nodes_explored > solutions.len());
    }

    // -----------------------------------------------------------------------
    // Solver: SolveStats.nodes_explored tracks work
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_stats_nodes_explored() {
        let net = two_var_network(
            Domain::new(vec![1.0, 2.0]),
            Domain::new(vec![1.0, 2.0]),
            ConstraintRelation::LessThan,
        );
        let config = CspConfig::default();
        let (_solutions, stats) = solve(&net, &config);
        assert!(stats.nodes_explored > 0);
    }

    // -----------------------------------------------------------------------
    // ConstraintRelation::reversed correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_constraint_relation_reversed() {
        // LessThan reversed → GreaterThan.
        // rev.holds(a, b) should be equivalent to original.holds(b, a).
        let rel = ConstraintRelation::LessThan;
        let rev = rel.reversed();

        // Original: 1 < 3 = true.
        assert!(rel.holds(1.0, 3.0));
        // Original: 3 < 1 = false.
        assert!(!rel.holds(3.0, 1.0));

        // Reversed (GreaterThan): rev.holds(a, b) ≡ original.holds(b, a).
        // rev.holds(3.0, 1.0) ≡ original.holds(1.0, 3.0) = 1 < 3 = true → GreaterThan(3,1) = true.
        assert!(rev.holds(3.0, 1.0));
        // rev.holds(1.0, 3.0) ≡ original.holds(3.0, 1.0) = 3 < 1 = false → GreaterThan(1,3) = false.
        assert!(!rev.holds(1.0, 3.0));

        // Verify LessOrEqual reversed → GreaterOrEqual.
        let loe = ConstraintRelation::LessOrEqual;
        let goe = loe.reversed();
        assert!(goe.holds(3.0, 1.0)); // 3 >= 1
        assert!(goe.holds(2.0, 2.0)); // 2 >= 2
        assert!(!goe.holds(1.0, 3.0)); // 1 >= 3 = false

        // Verify Equal reversed → Equal.
        let eq = ConstraintRelation::Equal;
        let eq_rev = eq.reversed();
        assert!(eq_rev.holds(5.0, 5.0));
        assert!(!eq_rev.holds(5.0, 6.0));

        // Verify NotEqual reversed → NotEqual.
        let ne = ConstraintRelation::NotEqual;
        let ne_rev = ne.reversed();
        assert!(ne_rev.holds(1.0, 2.0));
        assert!(!ne_rev.holds(1.0, 1.0));
    }

    // -----------------------------------------------------------------------
    // ConstraintNetwork: variable/constraint counts
    // -----------------------------------------------------------------------

    #[test]
    fn test_constraint_network_counts() {
        let mut net = ConstraintNetwork::new();
        net.add_variable("a", Domain::boolean());
        net.add_variable("b", Domain::boolean());
        net.add_constraint(BinaryConstraint::new(
            "a",
            "b",
            ConstraintRelation::NotEqual,
        ));
        assert_eq!(net.variable_count(), 2);
        assert_eq!(net.constraint_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Lexicographic ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_lexicographic_ordering() {
        let domains: HashMap<String, Domain> = [
            ("zebra".to_string(), Domain::new(vec![1.0])),
            ("apple".to_string(), Domain::new(vec![1.0, 2.0, 3.0])),
        ]
        .into_iter()
        .collect();
        let constraints: Vec<BinaryConstraint> = Vec::new();
        let unassigned = vec!["zebra".to_string(), "apple".to_string()];
        let chosen = select_variable(
            &unassigned,
            &domains,
            &constraints,
            &VarOrdering::Lexicographic,
        );
        assert_eq!(chosen, "apple");
    }
}
