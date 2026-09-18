//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::types::{
    AbstractState, AbstractSystem, BmcResult, CegarConfig, CegarResult, Concreteness,
    IntervalDomain, Predicate, RefinementStrategy, SafetyProperty, SpuriousCause, TransitionSystem,
};

/// State index type alias.
pub type StateIdx = usize;
/// Type alias for a boxed trace-level property predicate.
pub type TracePredicate = Box<dyn Fn(&[Vec<f64>]) -> bool>;
/// Verify energy conservation over a trajectory of states.
///
/// Each state is a `\[f64; 6\]` array of the form `\[x, y, z, vx, vy, vz\]`.
/// Energy is estimated as kinetic energy `0.5 * (vx² + vy² + vz²)` (unit mass).
///
/// Returns `true` if the energy variation across all states is within `tol`.
pub fn verify_energy_conservation(states: &[[f64; 6]], tol: f64) -> bool {
    if states.len() < 2 {
        return true;
    }
    let energies: Vec<f64> = states
        .iter()
        .map(|s| 0.5 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        .collect();
    let e0 = energies[0];
    energies.iter().all(|&e| (e - e0).abs() <= tol)
}
/// Verify linear momentum conservation over a trajectory of states.
///
/// Each state is `\[x, y, z, vx, vy, vz\]` (unit mass).
/// Returns `true` if the momentum vector changes by less than `tol` per
/// component.
pub fn verify_momentum_conservation(states: &[[f64; 6]], tol: f64) -> bool {
    if states.len() < 2 {
        return true;
    }
    let (px0, py0, pz0) = (states[0][3], states[0][4], states[0][5]);
    states.iter().all(|s| {
        (s[3] - px0).abs() <= tol && (s[4] - py0).abs() <= tol && (s[5] - pz0).abs() <= tol
    })
}
/// Verify angular momentum conservation.
///
/// Each state is `\[x, y, z, vx, vy, vz\]`.  Returns `true` if the z-component
/// of `r × v` changes by less than `tol`.
pub fn verify_angular_momentum_conservation(states: &[[f64; 6]], tol: f64) -> bool {
    if states.len() < 2 {
        return true;
    }
    let lz0 = states[0][0] * states[0][4] - states[0][1] * states[0][3];
    states
        .iter()
        .all(|s| ((s[0] * s[4] - s[1] * s[3]) - lz0).abs() <= tol)
}
/// Check Lyapunov stability of a fixed-point under a discrete map.
///
/// `states` is a trajectory; the function checks whether `‖state‖₂` is
/// non-increasing (on average) — a sufficient condition for Lyapunov stability.
pub fn check_lyapunov_stability(states: &[Vec<f64>]) -> bool {
    if states.len() < 2 {
        return true;
    }
    let norm = |s: &Vec<f64>| s.iter().map(|x| x * x).sum::<f64>().sqrt();
    let first_norm = norm(&states[0]);
    let last_norm = norm(states.last().expect("states has at least 2 entries"));
    last_norm <= first_norm * 1.001
}
/// A symbolic state: each variable holds an interval.
pub type SymbolicState = HashMap<String, IntervalDomain>;
/// Symbolic execution step: apply a numeric update `x := x + delta` to the
/// symbolic state, and check whether the result remains within `bounds`.
pub fn symbolic_step(
    state: &SymbolicState,
    var: &str,
    delta: f64,
    bounds: &IntervalDomain,
) -> Option<SymbolicState> {
    let iv = state.get(var).copied().unwrap_or_else(IntervalDomain::top);
    let next_iv = IntervalDomain::new(iv.lo + delta, iv.hi + delta);
    if next_iv.meet(bounds).is_non_empty() {
        let mut next = state.clone();
        next.insert(var.to_string(), next_iv.meet(bounds));
        Some(next)
    } else {
        None
    }
}
/// A concrete state in the transition system — the raw phase-space vector.
pub type ConcreteState = Vec<f64>;
/// Absolute deadline of one CEGAR iteration (`None` = no time limit).
type Deadline = Option<Instant>;
/// Smallest coordinate difference that is treated as a real difference when
/// refinement thresholds are derived from means or representatives.
const COORDINATE_TOLERANCE: f64 = 1e-12;
/// Number of expanded abstract states between two polls of the deadline.
const DEADLINE_POLL_INTERVAL: usize = 1024;
/// Deadline `timeout_ms` milliseconds from now; `0` means no limit.
///
/// A timeout so large that the deadline cannot be represented as an [`Instant`]
/// is treated as no limit.
fn deadline_after_millis(timeout_ms: u64) -> Deadline {
    if timeout_ms == 0 {
        None
    } else {
        Instant::now().checked_add(Duration::from_millis(timeout_ms))
    }
}
/// Whether `deadline` has been reached.
fn deadline_passed(deadline: Deadline) -> bool {
    deadline.is_some_and(|limit| Instant::now() >= limit)
}
/// A predicate-abstraction of `system` using `predicates`.
///
/// Each concrete state is mapped to an abstract state (a predicate bit-vector).
/// An abstract transition exists whenever a concrete transition exists between
/// the corresponding abstraction classes (existential abstraction), so every
/// concrete execution has an abstract counterpart and the abstraction
/// over-approximates the reachable behaviour.
///
/// Abstract states are numbered in order of first occurrence and the transition
/// list is sorted, so the whole analysis is deterministic.  Transitions that
/// mention a state outside the system do not exist and are ignored.
fn predicate_abstraction(system: &TransitionSystem, predicates: &[Predicate]) -> AbstractSystem {
    let mut abstract_states: Vec<AbstractState> = Vec::new();
    let mut class_of: Vec<usize> = Vec::with_capacity(system.model.states.len());
    let mut state_to_idx: HashMap<Vec<bool>, usize> = HashMap::new();
    for (ci, concrete) in system.model.states.iter().enumerate() {
        let bits: Vec<bool> = predicates.iter().map(|p| p.eval(concrete)).collect();
        let abs_idx = if let Some(&existing) = state_to_idx.get(&bits) {
            existing
        } else {
            let idx = abstract_states.len();
            abstract_states.push(AbstractState {
                bits: bits.clone(),
                concrete_idx: Some(ci),
            });
            state_to_idx.insert(bits, idx);
            idx
        };
        class_of.push(abs_idx);
    }
    let mut transitions: Vec<(usize, usize)> = system
        .model
        .transitions
        .iter()
        .filter_map(|&(from, to)| Some((*class_of.get(from)?, *class_of.get(to)?)))
        .collect();
    transitions.sort_unstable();
    transitions.dedup();
    AbstractSystem {
        states: abstract_states,
        transitions,
        class_of,
    }
}
/// Successor lists of the concrete system, indexed by state.
///
/// Transitions that mention a state outside `0..system.state_count()` do not
/// exist and are ignored, exactly as in [`predicate_abstraction`].
fn concrete_adjacency(system: &TransitionSystem) -> Vec<Vec<StateIdx>> {
    let mut adjacency: Vec<Vec<StateIdx>> = vec![Vec::new(); system.state_count()];
    for &(from, to) in &system.model.transitions {
        if to < adjacency.len()
            && let Some(successors) = adjacency.get_mut(from)
        {
            successors.push(to);
        }
    }
    adjacency
}
/// Abstract path from the root of a search tree to `end`, following `parent`.
fn abstract_path(parent: &[Option<usize>], end: usize) -> Vec<usize> {
    let mut path = vec![end];
    let mut current = end;
    while let Some(previous) = parent[current] {
        path.push(previous);
        current = previous;
    }
    path.reverse();
    path
}
/// Search the abstract system for a reachable abstract state that contains a
/// violating concrete state.
///
/// An abstract state is *bad* if ANY of the concrete states it represents
/// violates the property (a sound over-approximation of the bad set).  The search
/// is a breadth-first search from `init_abs` that expands the reachable abstract
/// graph level by level with a visited flag and a parent pointer per abstract
/// state, so a counterexample is the shortest abstract path to a bad state and
/// [`BmcResult::NoCounterexample`] is only returned once the whole reachable
/// abstract state space has been examined.
///
/// `bound` caps the number of levels that are expanded (`usize::MAX` = no cap).
/// If the cap is reached while the last level still has unvisited successors, the
/// search is incomplete and [`BmcResult::BoundReached`] is returned instead of a
/// verdict.  `deadline` is polled throughout; once it has passed the result is
/// [`BmcResult::TimedOut`].
///
/// `init_abs` must be a valid abstract state of `abs_sys`.
fn bounded_model_check(
    abs_sys: &AbstractSystem,
    system: &TransitionSystem,
    property: &SafetyProperty,
    init_abs: usize,
    bound: usize,
    deadline: Deadline,
) -> BmcResult {
    let mut bad = vec![false; abs_sys.states.len()];
    for (concrete_idx, concrete) in system.model.states.iter().enumerate() {
        if !property.check(concrete)
            && let Some(&class) = abs_sys.class_of.get(concrete_idx)
        {
            bad[class] = true;
        }
    }
    if deadline_passed(deadline) {
        return BmcResult::TimedOut;
    }
    if bad[init_abs] {
        return BmcResult::AbstractCex(vec![init_abs]);
    }
    let adjacency = abs_sys.adjacency();
    let mut visited = vec![false; abs_sys.states.len()];
    let mut parent: Vec<Option<usize>> = vec![None; abs_sys.states.len()];
    visited[init_abs] = true;
    let mut frontier: Vec<usize> = vec![init_abs];
    let mut depth = 0usize;
    let mut expanded = 0usize;
    while !frontier.is_empty() {
        if depth >= bound {
            let unexplored = frontier
                .iter()
                .any(|&state| adjacency[state].iter().any(|&succ| !visited[succ]));
            return if unexplored {
                BmcResult::BoundReached { bound }
            } else {
                BmcResult::NoCounterexample
            };
        }
        let mut next_frontier: Vec<usize> = Vec::new();
        for &state in &frontier {
            expanded += 1;
            if expanded.is_multiple_of(DEADLINE_POLL_INTERVAL) && deadline_passed(deadline) {
                return BmcResult::TimedOut;
            }
            for &succ in &adjacency[state] {
                if visited[succ] {
                    continue;
                }
                visited[succ] = true;
                parent[succ] = Some(state);
                if bad[succ] {
                    return BmcResult::AbstractCex(abstract_path(&parent, succ));
                }
                next_frontier.push(succ);
            }
        }
        if deadline_passed(deadline) {
            return BmcResult::TimedOut;
        }
        frontier = next_frontier;
        depth += 1;
    }
    BmcResult::NoCounterexample
}
/// Replay the abstract trace `abstract_trace` on the concrete `system`.
///
/// The trace is a path of abstract states that starts in the class of the
/// initial state and ends in an abstract state containing a violating concrete
/// state.  Because an abstract state merges many concrete states, one abstract
/// step may correspond to several concrete steps that stay inside the current
/// class before the transition into the next one.
///
/// The replay is a breadth-first search over pairs *(concrete state, trace
/// position)* that starts at the initial state at position 0.  A concrete
/// transition either stays at the current position (the successor lies in the
/// same class) or advances to the next one (the successor lies in the next class
/// of the trace).  The trace is *concrete* iff the search reaches a state that
/// violates the property at the last position; the returned trace is then a real
/// execution from the initial state: every consecutive pair of states is a
/// transition of `system` and the last state violates the property.
///
/// Otherwise the trace is *spurious* and the result records where the replay got
/// stuck, see [`SpuriousCause`].
///
/// `adjacency` is [`concrete_adjacency`] of `system`.
fn check_concreteness(
    system: &TransitionSystem,
    adjacency: &[Vec<StateIdx>],
    abs_sys: &AbstractSystem,
    property: &SafetyProperty,
    abstract_trace: &[usize],
) -> Concreteness {
    let positions = abstract_trace.len();
    let state_count = system.state_count();
    if positions == 0 || system.initial_idx >= state_count {
        return Concreteness::Spurious(SpuriousCause {
            reached: Vec::new(),
            needed: Vec::new(),
        });
    }
    let last = positions - 1;
    // Node `state * positions + position` of the product graph.
    let mut parent: Vec<Option<usize>> = vec![None; state_count * positions];
    let mut seen = vec![false; state_count * positions];
    let mut reached_at: Vec<Vec<StateIdx>> = vec![Vec::new(); positions];
    let start = system.initial_idx * positions;
    seen[start] = true;
    reached_at[0].push(system.initial_idx);
    let mut queue: VecDeque<usize> = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        let (state, position) = (node / positions, node % positions);
        if position == last && !property.check(&system.model.states[state]) {
            return Concreteness::Concrete(concrete_path(system, &parent, node, positions));
        }
        for &succ in &adjacency[state] {
            let class = abs_sys.class_of[succ];
            for next in [position, position + 1] {
                if next < positions && abstract_trace[next] == class {
                    let next_node = succ * positions + next;
                    if !seen[next_node] {
                        seen[next_node] = true;
                        parent[next_node] = Some(node);
                        reached_at[next].push(succ);
                        queue.push_back(next_node);
                    }
                }
            }
        }
    }
    // The positions that were reached form a prefix of the trace.
    let furthest = reached_at
        .iter()
        .rposition(|states| !states.is_empty())
        .unwrap_or(0);
    let class = abstract_trace[furthest];
    let needed: Vec<StateIdx> = (0..state_count)
        .filter(|&state| {
            abs_sys.class_of[state] == class
                && if furthest < last {
                    adjacency[state]
                        .iter()
                        .any(|&succ| abs_sys.class_of[succ] == abstract_trace[furthest + 1])
                } else {
                    !property.check(&system.model.states[state])
                }
        })
        .collect();
    Concreteness::Spurious(SpuriousCause {
        reached: std::mem::take(&mut reached_at[furthest]),
        needed,
    })
}
/// The concrete execution that ends in the product-graph node `goal`.
fn concrete_path(
    system: &TransitionSystem,
    parent: &[Option<usize>],
    goal: usize,
    positions: usize,
) -> Vec<ConcreteState> {
    let mut path = vec![system.model.states[goal / positions].clone()];
    let mut current = goal;
    while let Some(previous) = parent[current] {
        path.push(system.model.states[previous / positions].clone());
        current = previous;
    }
    path.reverse();
    path
}
/// Coordinate `var_idx` of `state`; missing coordinates read as `0.0`, the
/// convention of [`Predicate::eval`].
fn coordinate(state: &[f64], var_idx: usize) -> f64 {
    state.get(var_idx).copied().unwrap_or(0.0)
}
/// Mean of coordinate `var_idx` over `states` (`states` must not be empty).
fn mean_coordinate(states: &[&[f64]], var_idx: usize) -> f64 {
    states.iter().map(|s| coordinate(s, var_idx)).sum::<f64>() / states.len() as f64
}
/// Threshold `t` for a `>= t` predicate that puts the distinct values `a` and
/// `b` on different sides: their midpoint, or the larger value when the midpoint
/// rounds onto the smaller one.
fn separating_threshold(a: f64, b: f64) -> f64 {
    let (low, high) = if a <= b { (a, b) } else { (b, a) };
    let mid = low + (high - low) / 2.0;
    if low < mid && mid <= high { mid } else { high }
}
/// Whether `predicates` already contain a cut at `threshold` on `var_idx`.
///
/// `x >= t` and `x < t` are the same cut: they give complementary bits and the
/// same partition of the states.
fn is_known_cut(predicates: &[Predicate], var_idx: usize, threshold: f64) -> bool {
    predicates
        .iter()
        .any(|p| p.var_idx == var_idx && p.threshold.into_inner().to_bits() == threshold.to_bits())
}
/// Derive new predicates that separate the states the replay reached from the
/// states the abstract counterexample needs.
///
/// `cause.reached` and `cause.needed` are disjoint sets of concrete states that
/// the current abstraction cannot tell apart, so a counterexample that needs
/// the second set is spurious as long as the first set is all that is reachable.
/// The strategy proposes threshold cuts on single coordinates:
///
/// * [`RefinementStrategy::SyntaxGuided`] and
///   [`RefinementStrategy::InterpolationBased`] cut every coordinate halfway
///   between the mean of the reached states and the mean of the needed states
///   (when the means differ);
/// * [`RefinementStrategy::CraigInterpolant`] cuts every coordinate halfway
///   between a representative reached state and a representative needed state.
///
/// If the strategy finds no cut that is not yet part of `existing`, the
/// coordinates of the first reached state against all needed states (and of the
/// first needed state against all reached states) are searched for any
/// difference, which is then cut.  A cut that puts two states of one abstract
/// state on different sides strictly refines the abstraction, so every
/// refinement makes progress; the result is empty exactly when the two sets
/// cannot be separated by any coordinate threshold.
fn refine_predicates(
    system: &TransitionSystem,
    cause: &SpuriousCause,
    strategy: &RefinementStrategy,
    existing: &[Predicate],
) -> Vec<Predicate> {
    let reached: Vec<&[f64]> = cause
        .reached
        .iter()
        .filter_map(|&ci| system.model.states.get(ci).map(Vec::as_slice))
        .collect();
    let needed: Vec<&[f64]> = cause
        .needed
        .iter()
        .filter_map(|&ci| system.model.states.get(ci).map(Vec::as_slice))
        .collect();
    let (Some(&first_reached), Some(&first_needed)) = (reached.first(), needed.first()) else {
        return Vec::new();
    };
    let n_vars = reached
        .iter()
        .chain(needed.iter())
        .map(|s| s.len())
        .max()
        .unwrap_or(0);
    let mut cuts: Vec<(usize, f64)> = Vec::new();
    match strategy {
        RefinementStrategy::SyntaxGuided | RefinementStrategy::InterpolationBased => {
            for var_idx in 0..n_vars {
                let mean_reached = mean_coordinate(&reached, var_idx);
                let mean_needed = mean_coordinate(&needed, var_idx);
                if (mean_reached - mean_needed).abs() > COORDINATE_TOLERANCE {
                    cuts.push((var_idx, separating_threshold(mean_reached, mean_needed)));
                }
            }
        }
        RefinementStrategy::CraigInterpolant => {
            for var_idx in 0..n_vars {
                let value_reached = coordinate(first_reached, var_idx);
                let value_needed = coordinate(first_needed, var_idx);
                if (value_reached - value_needed).abs() > COORDINATE_TOLERANCE {
                    cuts.push((var_idx, separating_threshold(value_reached, value_needed)));
                }
            }
        }
    }
    cuts.retain(|&(var_idx, threshold)| {
        threshold.is_finite() && !is_known_cut(existing, var_idx, threshold)
    });
    if cuts.is_empty() {
        let differing = |reached_state: &[f64], needed_state: &[f64]| {
            (0..n_vars).find_map(|var_idx| {
                let a = coordinate(reached_state, var_idx);
                let b = coordinate(needed_state, var_idx);
                (a.is_finite() && b.is_finite() && a != b)
                    .then(|| (var_idx, separating_threshold(a, b)))
            })
        };
        let fallback = needed
            .iter()
            .find_map(|&q| differing(first_reached, q))
            .or_else(|| reached.iter().find_map(|&p| differing(p, first_needed)));
        if let Some((var_idx, threshold)) = fallback
            && !is_known_cut(existing, var_idx, threshold)
        {
            cuts.push((var_idx, threshold));
        }
    }
    cuts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    cuts.dedup_by(|a, b| a.0 == b.0 && a.1.to_bits() == b.1.to_bits());
    cuts.into_iter()
        .map(|(var_idx, threshold)| {
            Predicate::ge(
                format!("refine_v{var_idx}_ge_{threshold:.4}"),
                var_idx,
                threshold,
            )
        })
        .collect()
}
/// Compute initial predicates from the safety property by sampling state-space
/// boundaries implied by the property's structure.
///
/// Since `SafetyProperty` wraps an opaque closure, we derive boundary predicates
/// for each state variable based on the representative concrete states in `system`.
fn initial_predicates(system: &TransitionSystem, property: &SafetyProperty) -> Vec<Predicate> {
    let mut predicates: Vec<Predicate> = Vec::new();
    let (satisfying, violating): (Vec<&Vec<f64>>, Vec<&Vec<f64>>) =
        system.model.states.iter().partition(|s| property.check(s));
    if satisfying.is_empty() || violating.is_empty() {
        let n_vars = system.model.states.first().map(|s| s.len()).unwrap_or(1);
        for var_idx in 0..n_vars {
            let vals: Vec<f64> = system
                .model
                .states
                .iter()
                .map(|s| s.get(var_idx).copied().unwrap_or(0.0))
                .collect();
            if let (Some(&lo), Some(&hi)) = (
                vals.iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                vals.iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            ) && (hi - lo).abs() > 1e-12
            {
                let mid = (lo + hi) / 2.0;
                predicates.push(Predicate::ge(
                    format!("init_v{}_ge_{:.4}", var_idx, mid),
                    var_idx,
                    mid,
                ));
            }
        }
        return predicates;
    }
    let n_vars = system.model.states.first().map(|s| s.len()).unwrap_or(1);
    for var_idx in 0..n_vars {
        let max_sat = satisfying
            .iter()
            .map(|s| s.get(var_idx).copied().unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);
        let min_sat = satisfying
            .iter()
            .map(|s| s.get(var_idx).copied().unwrap_or(0.0))
            .fold(f64::INFINITY, f64::min);
        let max_vio = violating
            .iter()
            .map(|s| s.get(var_idx).copied().unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);
        let min_vio = violating
            .iter()
            .map(|s| s.get(var_idx).copied().unwrap_or(0.0))
            .fold(f64::INFINITY, f64::min);
        if min_sat > max_vio {
            let threshold = (min_sat + max_vio) / 2.0;
            predicates.push(Predicate::ge(
                format!("init_v{}_ge_{:.4}", var_idx, threshold),
                var_idx,
                threshold,
            ));
            predicates.push(Predicate::lt(
                format!("init_v{}_lt_{:.4}", var_idx, threshold),
                var_idx,
                threshold,
            ));
        } else if max_sat < min_vio {
            let threshold = (max_sat + min_vio) / 2.0;
            predicates.push(Predicate::ge(
                format!("init_v{}_ge_{:.4}", var_idx, threshold),
                var_idx,
                threshold,
            ));
            predicates.push(Predicate::lt(
                format!("init_v{}_lt_{:.4}", var_idx, threshold),
                var_idx,
                threshold,
            ));
        } else {
            let all_vals: Vec<f64> = satisfying
                .iter()
                .chain(violating.iter())
                .map(|s| s.get(var_idx).copied().unwrap_or(0.0))
                .collect();
            let lo = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (hi - lo).abs() > 1e-12 {
                let mid = (lo + hi) / 2.0;
                predicates.push(Predicate::ge(
                    format!("init_v{}_ge_{:.4}", var_idx, mid),
                    var_idx,
                    mid,
                ));
                predicates.push(Predicate::lt(
                    format!("init_v{}_lt_{:.4}", var_idx, mid),
                    var_idx,
                    mid,
                ));
            }
        }
    }
    predicates
}
/// Run the full CEGAR loop on `system`, verifying `property`.
///
/// Iterates:
/// 1. Compute the predicate abstraction from the current predicate set.
/// 2. Search the abstract system exhaustively (breadth first, see
///    [`CegarConfig::bmc_bound`] for the optional depth cap) for a reachable
///    abstract state that contains a violating concrete state.
/// 3. If there is none, the property holds in every reachable concrete state
///    (the abstraction over-approximates the reachable states) and the result is
///    [`CegarResult::Verified`].
/// 4. Otherwise replay the shortest abstract counterexample on the concrete
///    system.  If a real execution into a violating state exists, it is returned
///    as [`CegarResult::Violated`].
/// 5. If the counterexample is spurious, refine the predicates so that the
///    states the replay reached are separated from the states the counterexample
///    needs, and repeat.
///
/// The result is [`CegarResult::Unknown`] when the iteration budget
/// (`max_iterations`) is exhausted, when an iteration exceeds `timeout_ms`, when
/// the search depth cap is reached with abstract states left unexplored, when no
/// coordinate threshold can separate the states that refinement must tell apart,
/// or when the initial state is not a state of the system.  `Verified` and
/// `Violated` are never reported for an incomplete search.
///
/// Every refinement splits an abstract state, so for a finite system whose
/// states differ in their coordinates the loop decides after at most
/// `state_count() - 1` refinements, provided the budgets are large enough.
pub fn cegar_verify(
    system: &TransitionSystem,
    property: &SafetyProperty,
    config: &CegarConfig,
) -> CegarResult {
    cegar_verify_with_deadlines(system, property, config, &|| {
        deadline_after_millis(config.timeout_ms)
    })
}
/// [`cegar_verify`] with the per-iteration deadline supplied by `new_deadline`,
/// which is called once at the start of every iteration.
fn cegar_verify_with_deadlines(
    system: &TransitionSystem,
    property: &SafetyProperty,
    config: &CegarConfig,
    new_deadline: &dyn Fn() -> Deadline,
) -> CegarResult {
    if system.initial_idx >= system.state_count() {
        return CegarResult::Unknown {
            reason: format!(
                "Initial state {} is not a state of the system ({} states)",
                system.initial_idx,
                system.state_count()
            ),
        };
    }
    let adjacency = concrete_adjacency(system);
    let timeout = |iteration: usize| CegarResult::Unknown {
        reason: format!(
            "Time limit exceeded in iteration {iteration} (timeout_ms = {})",
            config.timeout_ms
        ),
    };
    let mut predicates = initial_predicates(system, property);
    for iteration in 0..config.max_iterations {
        let deadline = new_deadline();
        let abstract_sys = predicate_abstraction(system, &predicates);
        if deadline_passed(deadline) {
            return timeout(iteration);
        }
        let init_abs = abstract_sys.class_of[system.initial_idx];
        let bmc_result = bounded_model_check(
            &abstract_sys,
            system,
            property,
            init_abs,
            config.bmc_bound,
            deadline,
        );
        match bmc_result {
            BmcResult::NoCounterexample => return CegarResult::Verified,
            BmcResult::TimedOut => return timeout(iteration),
            BmcResult::BoundReached { bound } => {
                return CegarResult::Unknown {
                    reason: format!(
                        "Search depth bound of {bound} reached in iteration {iteration} \
                         with unexplored abstract states"
                    ),
                };
            }
            BmcResult::AbstractCex(abstract_trace) => {
                match check_concreteness(
                    system,
                    &adjacency,
                    &abstract_sys,
                    property,
                    &abstract_trace,
                ) {
                    Concreteness::Concrete(concrete_trace) => {
                        return CegarResult::Violated {
                            trace: concrete_trace,
                        };
                    }
                    Concreteness::Spurious(cause) => {
                        if deadline_passed(deadline) {
                            return timeout(iteration);
                        }
                        let new_preds = refine_predicates(
                            system,
                            &cause,
                            &config.refinement_strategy,
                            &predicates,
                        );
                        if new_preds.is_empty() {
                            return CegarResult::Unknown {
                                reason: format!(
                                    "Refinement stalled at iteration {iteration} (no coordinate \
                                     threshold separates the states of the spurious counterexample)"
                                ),
                            };
                        }
                        predicates.extend(new_preds);
                    }
                }
            }
        }
    }
    CegarResult::Unknown {
        reason: format!(
            "Exceeded {} iterations without convergence",
            config.max_iterations
        ),
    }
}
/// Verify a collection of named properties against a simulation trace.
///
/// Returns a map from property name to `true`/`false`.
pub fn verify_trace_properties(
    trace: &[Vec<f64>],
    properties: &[(&str, TracePredicate)],
) -> HashMap<String, bool> {
    properties
        .iter()
        .map(|(name, predicate)| (name.to_string(), predicate(trace)))
        .collect()
}
/// Return `true` if `predicate` holds for every element of `trace`.
pub fn always<S, F: Fn(&S) -> bool>(trace: &[S], predicate: F) -> bool {
    trace.iter().all(predicate)
}
/// Return `true` if `predicate` holds for at least one element of `trace`.
pub fn eventually<S, F: Fn(&S) -> bool>(trace: &[S], predicate: F) -> bool {
    trace.iter().any(predicate)
}
/// Return `true` if `predicate` holds at every position strictly after
/// a position where `trigger` holds.
pub fn globally_after<S, F: Fn(&S) -> bool, G: Fn(&S) -> bool>(
    trace: &[S],
    trigger: F,
    predicate: G,
) -> bool {
    let mut triggered = false;
    for s in trace {
        if triggered && !predicate(s) {
            return false;
        }
        if trigger(s) {
            triggered = true;
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::*;
    #[test]
    fn test_interval_contains() {
        let iv = IntervalDomain::new(1.0, 3.0);
        assert!(iv.contains(2.0));
        assert!(!iv.contains(4.0));
    }
    #[test]
    fn test_interval_join() {
        let a = IntervalDomain::new(1.0, 3.0);
        let b = IntervalDomain::new(2.0, 5.0);
        let j = a.join(&b);
        assert_eq!(j.lo, 1.0);
        assert_eq!(j.hi, 5.0);
    }
    #[test]
    fn test_interval_meet() {
        let a = IntervalDomain::new(1.0, 4.0);
        let b = IntervalDomain::new(2.0, 6.0);
        let m = a.meet(&b);
        assert_eq!(m.lo, 2.0);
        assert_eq!(m.hi, 4.0);
    }
    #[test]
    fn test_interval_add() {
        let a = IntervalDomain::new(1.0, 2.0);
        let b = IntervalDomain::new(3.0, 4.0);
        let c = a.add(&b);
        assert_eq!(c.lo, 4.0);
        assert_eq!(c.hi, 6.0);
    }
    #[test]
    fn test_interval_sub() {
        let a = IntervalDomain::new(3.0, 5.0);
        let b = IntervalDomain::new(1.0, 2.0);
        let c = a.sub(&b);
        assert_eq!(c.lo, 1.0);
        assert_eq!(c.hi, 4.0);
    }
    #[test]
    fn test_interval_mul() {
        let a = IntervalDomain::new(2.0, 3.0);
        let b = IntervalDomain::new(4.0, 5.0);
        let c = a.mul(&b);
        assert_eq!(c.lo, 8.0);
        assert_eq!(c.hi, 15.0);
    }
    #[test]
    fn test_interval_div_no_zero() {
        let a = IntervalDomain::new(6.0, 8.0);
        let b = IntervalDomain::new(2.0, 4.0);
        let c = a.div(&b);
        assert!(c.lo >= 1.5 - 1e-9);
        assert!(c.hi <= 4.0 + 1e-9);
    }
    #[test]
    fn test_interval_div_contains_zero() {
        let a = IntervalDomain::new(1.0, 2.0);
        let b = IntervalDomain::new(-1.0, 1.0);
        let c = a.div(&b);
        assert_eq!(c, IntervalDomain::top());
    }
    #[test]
    fn test_interval_width() {
        let iv = IntervalDomain::new(2.0, 5.0);
        assert!((iv.width() - 3.0).abs() < 1e-10);
    }
    #[test]
    fn test_ltl_check_safety() {
        let ltl = LinearTemporalLogic::new("always x >= 0");
        let trace = vec![1.0_f64, 2.0, 3.0];
        assert!(ltl.check_safety(&trace, |&x| x >= 0.0));
        let bad_trace = vec![1.0_f64, -1.0, 3.0];
        assert!(!ltl.check_safety(&bad_trace, |&x| x >= 0.0));
    }
    #[test]
    fn test_ltl_check_liveness() {
        let ltl = LinearTemporalLogic::new("eventually x > 10");
        let trace = vec![1.0_f64, 5.0, 11.0];
        assert!(ltl.check_liveness(&trace, |&x| x > 10.0));
        let bad = vec![1.0_f64, 2.0, 3.0];
        assert!(!ltl.check_liveness(&bad, |&x| x > 10.0));
    }
    #[test]
    fn test_ltl_check_next() {
        let ltl = LinearTemporalLogic::new("next");
        let trace = vec![true, true, false];
        assert!(!ltl.check_next(&trace, |&x| x, |&x| x));
    }
    #[test]
    fn test_ltl_check_until() {
        let ltl = LinearTemporalLogic::new("until");
        let trace = vec![1i32, 1, 1, 2];
        assert!(ltl.check_until(&trace, |&x| x == 1, |&x| x == 2));
        let bad = vec![1i32, 0, 2];
        assert!(!ltl.check_until(&bad, |&x| x == 1, |&x| x == 2));
    }
    fn simple_model() -> ModelChecker {
        ModelChecker::new(vec![vec![0.0], vec![1.0], vec![2.0]], vec![(0, 1), (1, 2)])
    }
    #[test]
    fn test_model_reachable_states() {
        let mc = simple_model();
        let r = mc.reachable_states(0);
        assert!(r.contains(&0));
        assert!(r.contains(&1));
        assert!(r.contains(&2));
    }
    #[test]
    fn test_model_satisfies_invariant() {
        let mc = simple_model();
        assert!(mc.satisfies_invariant(0, |s| s[0] >= 0.0));
        assert!(!mc.satisfies_invariant(0, |s| s[0] > 1.5));
    }
    #[test]
    fn test_model_find_counterexample() {
        let mc = simple_model();
        let cex = mc.find_counterexample(0, |s| s[0] < 2.0);
        assert_eq!(cex, Some(2));
    }
    #[test]
    fn test_model_shortest_path() {
        let mc = simple_model();
        let path = mc.shortest_path(0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }
    #[test]
    fn test_abstract_interp_bind_get() {
        let mut ai = AbstractInterpretation::new();
        ai.bind("x", 3.0);
        assert_eq!(ai.get("x"), IntervalDomain::new(3.0, 3.0));
    }
    #[test]
    fn test_abstract_interp_widening() {
        let ai = AbstractInterpretation::new();
        let prev = IntervalDomain::new(0.0, 1.0);
        let next = IntervalDomain::new(0.0, 2.0);
        let w = ai.widening(&prev, &next);
        assert_eq!(w.hi, f64::INFINITY);
    }
    #[test]
    fn test_abstract_interp_narrowing() {
        let ai = AbstractInterpretation::new();
        let prev = IntervalDomain::new(f64::NEG_INFINITY, f64::INFINITY);
        let next = IntervalDomain::new(-5.0, 5.0);
        let n = ai.narrowing(&prev, &next);
        assert_eq!(n.lo, -5.0);
        assert_eq!(n.hi, 5.0);
    }
    #[test]
    fn test_abstract_interp_is_non_negative() {
        let mut ai = AbstractInterpretation::new();
        ai.bind_interval("x", IntervalDomain::new(0.0, 10.0));
        assert!(ai.is_non_negative("x"));
        ai.bind_interval("y", IntervalDomain::new(-1.0, 10.0));
        assert!(!ai.is_non_negative("y"));
    }
    #[test]
    fn test_bisimulation_same_state() {
        let mut lts = LabeledTransitionSystem::new(2);
        lts.add_label(0, "a");
        lts.add_label(1, "a");
        lts.add_transition(0, "x", 0);
        lts.add_transition(1, "x", 1);
        let checker = BisimulationChecker::new(lts);
        assert!(checker.are_bisimilar(0, 1));
    }
    #[test]
    fn test_bisimulation_different_labels() {
        let mut lts = LabeledTransitionSystem::new(2);
        lts.add_label(0, "a");
        lts.add_label(1, "b");
        let checker = BisimulationChecker::new(lts);
        assert!(!checker.are_bisimilar(0, 1));
    }
    #[test]
    fn test_bisimulation_quotient_size() {
        let mut lts = LabeledTransitionSystem::new(3);
        lts.add_label(0, "a");
        lts.add_label(1, "a");
        lts.add_label(2, "b");
        let checker = BisimulationChecker::new(lts);
        assert_eq!(checker.quotient_size(), 2);
    }
    #[test]
    fn test_verify_energy_conservation_ok() {
        let states = vec![[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; 5];
        assert!(verify_energy_conservation(&states, 1e-9));
    }
    #[test]
    fn test_verify_energy_conservation_fail() {
        let s1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let s2 = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert!(!verify_energy_conservation(&[s1, s2], 1e-9));
    }
    #[test]
    fn test_verify_momentum_conservation_ok() {
        let states = vec![[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]; 4];
        assert!(verify_momentum_conservation(&states, 1e-9));
    }
    #[test]
    fn test_verify_momentum_conservation_fail() {
        let s1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let s2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!verify_momentum_conservation(&[s1, s2], 1e-9));
    }
    #[test]
    fn test_sat_trivial_sat() {
        let mut sat = SatisfiabilityChecker::new(1);
        sat.add_clause(vec![Literal::pos(0)]);
        let result = sat.solve();
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert!(sat.check_assignment(&assignment));
    }
    #[test]
    fn test_sat_trivial_unsat() {
        let mut sat = SatisfiabilityChecker::new(1);
        sat.add_clause(vec![Literal::pos(0)]);
        sat.add_clause(vec![Literal::neg(0)]);
        assert!(sat.solve().is_none());
    }
    #[test]
    fn test_sat_two_vars() {
        let mut sat = SatisfiabilityChecker::new(2);
        sat.add_clause(vec![Literal::pos(0), Literal::pos(1)]);
        sat.add_clause(vec![Literal::neg(0), Literal::pos(1)]);
        let result = sat.solve();
        assert!(result.is_some());
        assert!(sat.check_assignment(&result.unwrap()));
    }
    #[test]
    fn test_sat_check_assignment() {
        let mut sat = SatisfiabilityChecker::new(2);
        sat.add_clause(vec![Literal::pos(0), Literal::pos(1)]);
        assert!(sat.check_assignment(&[false, true]));
        assert!(!sat.check_assignment(&[false, false]));
    }
    #[test]
    fn test_type_state_fire_ok() {
        let mut proto = TypeStateProtocol::new("idle");
        proto.add_transition("idle", "start", "running");
        proto.add_transition("running", "stop", "idle");
        assert!(proto.fire("start").is_ok());
        assert_eq!(proto.current_state, "running");
    }
    #[test]
    fn test_type_state_fire_err() {
        let mut proto = TypeStateProtocol::new("idle");
        proto.add_transition("idle", "start", "running");
        assert!(proto.fire("stop").is_err());
    }
    #[test]
    fn test_type_state_is_reachable() {
        let mut proto = TypeStateProtocol::new("idle");
        proto.add_transition("idle", "start", "running");
        proto.add_transition("running", "pause", "paused");
        assert!(proto.is_reachable("idle", "paused"));
        assert!(!proto.is_reachable("idle", "done"));
    }
    #[test]
    fn test_always() {
        assert!(always(&[1, 2, 3], |&x| x > 0));
        assert!(!always(&[1, -1, 3], |&x| x > 0));
    }
    #[test]
    fn test_eventually() {
        assert!(eventually(&[0, 0, 5], |&x| x > 4));
        assert!(!eventually(&[0, 0, 0], |&x| x > 4));
    }
    #[test]
    fn test_globally_after() {
        let trace = vec![0i32, 1, 2, 3];
        assert!(globally_after(&trace, |&x| x == 1, |&x| x > 0));
    }
    #[test]
    fn test_lyapunov_stable() {
        let trace: Vec<Vec<f64>> = (0..5).map(|i| vec![1.0 / (i as f64 + 1.0)]).collect();
        assert!(check_lyapunov_stability(&trace));
    }
    fn simple_safe_system() -> TransitionSystem {
        let mc = ModelChecker::new(vec![vec![0.0], vec![1.0], vec![2.0]], vec![(0, 1), (1, 2)]);
        TransitionSystem::new(mc, 0)
    }
    fn simple_violated_system() -> TransitionSystem {
        let mc = ModelChecker::new(vec![vec![0.0], vec![-1.0]], vec![(0, 1)]);
        TransitionSystem::new(mc, 0)
    }
    /// Index of the state with exactly these coordinates (the test systems give
    /// every state distinct coordinates).
    fn state_index(system: &TransitionSystem, state: &[f64]) -> Option<StateIdx> {
        system
            .model
            .states
            .iter()
            .position(|s| s.as_slice() == state)
    }
    /// `trace` must be a real execution: it starts in the initial state, every
    /// consecutive pair is a transition of `system`, and the last state violates
    /// `property`.
    fn assert_real_violating_execution(
        system: &TransitionSystem,
        property: &SafetyProperty,
        trace: &[ConcreteState],
    ) {
        let indices: Vec<StateIdx> = trace
            .iter()
            .map(|s| state_index(system, s).expect("trace state belongs to the system"))
            .collect();
        assert_eq!(
            indices.first().copied(),
            Some(system.initial_idx),
            "trace must start in the initial state: {trace:?}"
        );
        for pair in indices.windows(2) {
            assert!(
                system.successors(pair[0]).contains(&pair[1]),
                "{} -> {} is not a transition: {trace:?}",
                pair[0],
                pair[1]
            );
        }
        let last = trace.last().expect("trace is not empty");
        assert!(
            !property.check(last),
            "trace must end in a violating state: {trace:?}"
        );
    }
    /// Chain `0 -> 1 -> ... -> n-1` of `n >= 2` states in thermometer code: state
    /// `i` has the `n - 1` coordinates `x_k = 1` if `i > k` and `0` otherwise.
    /// Every coordinate is separated at `0.5` by the initial predicates, so each
    /// state is its own abstract state and the abstract chain is `n - 1` steps long.
    fn thermometer_chain(n: usize) -> TransitionSystem {
        let states: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n - 1).map(|k| if i > k { 1.0 } else { 0.0 }).collect())
            .collect();
        let transitions: Vec<(StateIdx, StateIdx)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        TransitionSystem::new(ModelChecker::new(states, transitions), 0)
    }
    /// Only the last state of `thermometer_chain(n)` violates this property.
    fn thermometer_property(n: usize) -> SafetyProperty {
        SafetyProperty::new("last coordinate below one half", move |s| s[n - 2] < 0.5)
    }
    /// Identity abstraction: one abstract state per concrete state.
    fn identity_abstraction(system: &TransitionSystem) -> AbstractSystem {
        let mut transitions = system.model.transitions.clone();
        transitions.sort_unstable();
        AbstractSystem {
            states: (0..system.state_count())
                .map(|i| AbstractState {
                    bits: vec![i % 2 == 0],
                    concrete_idx: Some(i),
                })
                .collect(),
            transitions,
            class_of: (0..system.state_count()).collect(),
        }
    }
    fn scalar_chain(n: usize) -> TransitionSystem {
        let states: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let transitions: Vec<(StateIdx, StateIdx)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
        TransitionSystem::new(ModelChecker::new(states, transitions), 0)
    }
    #[test]
    fn test_cegar_simple_safe_property() {
        let system = simple_safe_system();
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig::default();
        let result = cegar_verify(&system, &property, &config);
        assert!(
            matches!(result, CegarResult::Verified),
            "Expected Verified, got {:?}",
            result
        );
    }
    #[test]
    fn test_cegar_simple_violated_property() {
        let system = simple_violated_system();
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig::default();
        let result = cegar_verify(&system, &property, &config);
        assert!(
            matches!(result, CegarResult::Violated { .. }),
            "Expected Violated, got {:?}",
            result
        );
        if let CegarResult::Violated { trace } = &result {
            assert!(!trace.is_empty(), "Trace should not be empty");
            assert_real_violating_execution(&system, &property, trace);
        }
    }
    #[test]
    fn test_cegar_refinement_finds_the_violation_behind_a_merged_state() {
        let mc = ModelChecker::new(
            vec![vec![5.0], vec![10.0], vec![-1.0]],
            vec![(0, 1), (1, 2)],
        );
        let system = TransitionSystem::new(mc, 0);
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig {
            max_iterations: 10,
            bmc_bound: 5,
            ..CegarConfig::default()
        };
        match cegar_verify(&system, &property, &config) {
            CegarResult::Violated { trace } => {
                assert_eq!(trace, vec![vec![5.0], vec![10.0], vec![-1.0]]);
                assert_real_violating_execution(&system, &property, &trace);
            }
            other => panic!("Expected Violated, got {other:?}"),
        }
    }
    #[test]
    fn test_cegar_zero_iteration_budget_returns_unknown() {
        let mc = ModelChecker::new(vec![vec![1.0], vec![-1.0]], vec![(0, 1)]);
        let system = TransitionSystem::new(mc, 0);
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig {
            max_iterations: 0,
            ..CegarConfig::default()
        };
        let result = cegar_verify(&system, &property, &config);
        assert!(
            matches!(result, CegarResult::Unknown { .. }),
            "Expected Unknown with max_iterations=0, got {:?}",
            result
        );
    }
    #[test]
    fn test_cegar_violated_trace_is_the_shortest_real_execution() {
        let n = 6usize;
        let system = scalar_chain(n);
        let property = SafetyProperty::new("x < 5", |s| s[0] < 5.0);
        let result = cegar_verify(&system, &property, &CegarConfig::default());
        let CegarResult::Violated { trace } = result else {
            panic!("Expected Violated result, got {result:?}");
        };
        assert_eq!(trace.len(), n);
        assert_real_violating_execution(&system, &property, &trace);
    }
    #[test]
    fn test_cegar_craig_interpolant_strategy() {
        let system = simple_violated_system();
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig {
            refinement_strategy: RefinementStrategy::CraigInterpolant,
            ..CegarConfig::default()
        };
        let result = cegar_verify(&system, &property, &config);
        assert!(
            matches!(result, CegarResult::Violated { .. }),
            "Expected Violated with CraigInterpolant strategy, got {:?}",
            result
        );
    }
    #[test]
    fn test_cegar_scalar_chain_with_violation_beyond_bound_is_violated() {
        // 12 states, only the last one violates: found although the cap equals 10.
        let n = 12usize;
        let system = scalar_chain(n);
        let property = SafetyProperty::new("x < 11", |s| s[0] < 11.0);
        let config = CegarConfig {
            bmc_bound: 10,
            ..CegarConfig::default()
        };
        let CegarResult::Violated { trace } = cegar_verify(&system, &property, &config) else {
            panic!("Expected Violated");
        };
        assert_eq!(trace.len(), n);
        assert_real_violating_execution(&system, &property, &trace);
    }
    #[test]
    fn test_cegar_deep_abstract_chain_is_not_verified_beyond_the_old_bound() {
        // The bad abstract state is 11 transitions away.  A search that stopped
        // at depth 10 used to report `Verified` for this reachable violation.
        let n = 12usize;
        let system = thermometer_chain(n);
        let property = thermometer_property(n);
        let CegarResult::Violated { trace } =
            cegar_verify(&system, &property, &CegarConfig::default())
        else {
            panic!("Expected Violated");
        };
        assert_eq!(trace.len(), n);
        assert_eq!(trace, system.model.states);
        assert_real_violating_execution(&system, &property, &trace);
    }
    #[test]
    fn test_cegar_depth_bound_reached_is_unknown_never_verified() {
        let n = 12usize;
        let system = thermometer_chain(n);
        let property = thermometer_property(n);
        for bmc_bound in [0usize, 1, 5, 10] {
            let config = CegarConfig {
                bmc_bound,
                ..CegarConfig::default()
            };
            match cegar_verify(&system, &property, &config) {
                CegarResult::Unknown { reason } => {
                    assert!(reason.contains("depth bound"), "{reason}");
                }
                other => panic!("bound {bmc_bound}: expected Unknown, got {other:?}"),
            }
        }
        // The bad abstract state is exactly 11 transitions deep.
        let config = CegarConfig {
            bmc_bound: 11,
            ..CegarConfig::default()
        };
        assert!(matches!(
            cegar_verify(&system, &property, &config),
            CegarResult::Violated { .. }
        ));
    }
    #[test]
    fn test_cegar_safe_deep_chain_is_verified_only_by_a_complete_search() {
        let n = 12usize;
        let system = thermometer_chain(n);
        let always = SafetyProperty::new("always", |_| true);
        assert!(matches!(
            cegar_verify(&system, &always, &CegarConfig::default()),
            CegarResult::Verified
        ));
        for bmc_bound in [0usize, 5, 10] {
            let config = CegarConfig {
                bmc_bound,
                ..CegarConfig::default()
            };
            assert!(
                matches!(
                    cegar_verify(&system, &always, &config),
                    CegarResult::Unknown { .. }
                ),
                "bound {bmc_bound} leaves abstract states unexplored"
            );
        }
        // With the cap at the depth of the last abstract state nothing is left.
        let config = CegarConfig {
            bmc_bound: n - 1,
            ..CegarConfig::default()
        };
        assert!(matches!(
            cegar_verify(&system, &always, &config),
            CegarResult::Verified
        ));
    }
    #[test]
    fn test_cegar_zero_depth_bound_does_not_verify() {
        let system = simple_violated_system();
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig {
            bmc_bound: 0,
            ..CegarConfig::default()
        };
        match cegar_verify(&system, &property, &config) {
            CegarResult::Unknown { reason } => assert!(reason.contains("depth bound"), "{reason}"),
            other => panic!("Expected Unknown, got {other:?}"),
        }
    }
    #[test]
    fn test_bounded_model_check_cap_semantics() {
        let n = 12usize;
        let system = scalar_chain(n);
        let abstraction = identity_abstraction(&system);
        let property = SafetyProperty::new("x < 11", |s| s[0] < 11.0);
        let full_path: Vec<usize> = (0..n).collect();
        let check =
            |bound: usize| bounded_model_check(&abstraction, &system, &property, 0, bound, None);
        assert!(matches!(check(usize::MAX), BmcResult::AbstractCex(p) if p == full_path));
        assert!(matches!(check(11), BmcResult::AbstractCex(p) if p == full_path));
        assert!(matches!(check(10), BmcResult::BoundReached { bound: 10 }));
        assert!(matches!(check(0), BmcResult::BoundReached { bound: 0 }));
    }
    #[test]
    fn test_bounded_model_check_terminates_on_cycles_and_proves_closed_searches() {
        let n = 12usize;
        let mut system = scalar_chain(n);
        system.model.transitions.push((n - 1, 0));
        system.model.transitions.push((3, 3));
        let abstraction = identity_abstraction(&system);
        let always = SafetyProperty::new("always", |_| true);
        let check =
            |bound: usize| bounded_model_check(&abstraction, &system, &always, 0, bound, None);
        assert!(matches!(check(usize::MAX), BmcResult::NoCounterexample));
        // The last level only leads back to visited states: nothing is left.
        assert!(matches!(check(n - 1), BmcResult::NoCounterexample));
        assert!(matches!(check(n - 2), BmcResult::BoundReached { .. }));
    }
    #[test]
    fn test_bounded_model_check_returns_a_shortest_path() {
        let mut system = scalar_chain(5);
        system.model.transitions.push((0, 4));
        let abstraction = identity_abstraction(&system);
        let property = SafetyProperty::new("x < 4", |s| s[0] < 4.0);
        let result = bounded_model_check(&abstraction, &system, &property, 0, usize::MAX, None);
        assert!(matches!(result, BmcResult::AbstractCex(p) if p == vec![0, 4]));
    }
    #[test]
    fn test_bounded_model_check_expired_deadline_times_out() {
        let system = scalar_chain(4);
        let abstraction = identity_abstraction(&system);
        let property = SafetyProperty::new("x < 3", |s| s[0] < 3.0);
        let expired = Some(Instant::now());
        let result = bounded_model_check(&abstraction, &system, &property, 0, usize::MAX, expired);
        assert!(matches!(result, BmcResult::TimedOut), "{result:?}");
        let result = bounded_model_check(&abstraction, &system, &property, 0, usize::MAX, None);
        assert!(matches!(result, BmcResult::AbstractCex(_)), "{result:?}");
    }
    #[test]
    fn test_cegar_expired_deadline_returns_unknown() {
        let system = simple_violated_system();
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let config = CegarConfig::default();
        let expired =
            cegar_verify_with_deadlines(&system, &property, &config, &|| Some(Instant::now()));
        match expired {
            CegarResult::Unknown { reason } => assert!(reason.contains("Time limit"), "{reason}"),
            other => panic!("Expected Unknown, got {other:?}"),
        }
        let unlimited = cegar_verify_with_deadlines(&system, &property, &config, &|| None);
        assert!(matches!(unlimited, CegarResult::Violated { .. }));
    }
    #[test]
    fn test_cegar_timeout_ms_limits_each_iteration() {
        // Evaluating the property takes at least 20 ms, far beyond the 1 ms limit.
        let system = simple_violated_system();
        let slow_property = SafetyProperty::new("slow", |s| {
            std::thread::sleep(Duration::from_millis(20));
            s[0] >= 0.0
        });
        let limited = CegarConfig {
            timeout_ms: 1,
            ..CegarConfig::default()
        };
        match cegar_verify(&system, &slow_property, &limited) {
            CegarResult::Unknown { reason } => assert!(reason.contains("Time limit"), "{reason}"),
            other => panic!("Expected Unknown, got {other:?}"),
        }
        // Generous limits, including one that overflows `Instant`, do not interfere.
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        for timeout_ms in [0, 60_000, u64::MAX] {
            let config = CegarConfig {
                timeout_ms,
                ..CegarConfig::default()
            };
            assert!(
                matches!(
                    cegar_verify(&system, &property, &config),
                    CegarResult::Violated { .. }
                ),
                "timeout_ms = {timeout_ms}"
            );
        }
    }
    #[test]
    fn test_cegar_initial_state_outside_the_system_is_unknown() {
        let states = vec![vec![0.0], vec![1.0], vec![2.0]];
        let transitions = vec![(0, 1), (1, 2)];
        let property = SafetyProperty::new("always", |_| true);
        for initial in [3usize, 100] {
            let system = TransitionSystem::new(
                ModelChecker::new(states.clone(), transitions.clone()),
                initial,
            );
            match cegar_verify(&system, &property, &CegarConfig::default()) {
                CegarResult::Unknown { reason } => {
                    assert!(reason.contains("Initial state"), "{reason}");
                }
                other => panic!("initial {initial}: expected Unknown, got {other:?}"),
            }
        }
        let empty = TransitionSystem::new(ModelChecker::new(Vec::new(), Vec::new()), 0);
        assert!(matches!(
            cegar_verify(&empty, &property, &CegarConfig::default()),
            CegarResult::Unknown { .. }
        ));
    }
    #[test]
    fn test_cegar_ignores_transitions_outside_the_system() {
        let mc = ModelChecker::new(
            vec![vec![0.0], vec![1.0]],
            vec![(0, 1), (1, 9), (7, 0), (9, 9)],
        );
        let system = TransitionSystem::new(mc, 0);
        let property = SafetyProperty::new("x >= 0", |s| s[0] >= 0.0);
        let result = cegar_verify(&system, &property, &CegarConfig::default());
        assert!(matches!(result, CegarResult::Verified), "{result:?}");
    }
    /// Three states `x = 0, 1, 2` with the single transition `0 -> 2`.  Only
    /// `x = 1` violates the property and it is unreachable, but the initial
    /// predicates put it into the same abstract state as the reachable `x = 2`.
    fn merged_state_system(transitions: Vec<(StateIdx, StateIdx)>) -> TransitionSystem {
        let mc = ModelChecker::new(vec![vec![0.0], vec![1.0], vec![2.0]], transitions);
        TransitionSystem::new(mc, 0)
    }
    const ALL_STRATEGIES: [RefinementStrategy; 3] = [
        RefinementStrategy::SyntaxGuided,
        RefinementStrategy::CraigInterpolant,
        RefinementStrategy::InterpolationBased,
    ];
    #[test]
    fn test_cegar_merged_abstract_state_with_unreachable_violation_is_verified() {
        let system = merged_state_system(vec![(0, 2)]);
        let property = SafetyProperty::new("away from one", |s| (s[0] - 1.0).abs() > 0.5);
        for refinement_strategy in ALL_STRATEGIES {
            let config = CegarConfig {
                refinement_strategy,
                ..CegarConfig::default()
            };
            let result = cegar_verify(&system, &property, &config);
            assert!(
                matches!(result, CegarResult::Verified),
                "{refinement_strategy:?}: a trace ending in the satisfying state x = 2 \
                 is not a counterexample, got {result:?}"
            );
        }
    }
    #[test]
    fn test_cegar_merged_abstract_state_with_reachable_violation_gives_a_real_trace() {
        let system = merged_state_system(vec![(0, 2), (2, 1)]);
        let property = SafetyProperty::new("away from one", |s| (s[0] - 1.0).abs() > 0.5);
        for refinement_strategy in ALL_STRATEGIES {
            let config = CegarConfig {
                refinement_strategy,
                ..CegarConfig::default()
            };
            match cegar_verify(&system, &property, &config) {
                CegarResult::Violated { trace } => {
                    assert_eq!(trace, vec![vec![0.0], vec![2.0], vec![1.0]]);
                    assert_real_violating_execution(&system, &property, &trace);
                }
                other => panic!("{refinement_strategy:?}: expected Violated, got {other:?}"),
            }
        }
    }
    #[test]
    fn test_cegar_spurious_crossing_is_refined_away_by_every_strategy() {
        // The initial predicates merge the initial state 0 with state 1, and only
        // state 1 can step into the violating state 2.  The abstract trace
        // {0, 1} -> {2} is spurious because the replay gets stuck in state 0.
        let mc = ModelChecker::new(vec![vec![0.0], vec![0.5], vec![1.0]], vec![(1, 2)]);
        let system = TransitionSystem::new(mc, 0);
        let property = SafetyProperty::new("x < 1", |s| s[0] < 1.0);
        for refinement_strategy in ALL_STRATEGIES {
            let config = CegarConfig {
                refinement_strategy,
                ..CegarConfig::default()
            };
            let result = cegar_verify(&system, &property, &config);
            assert!(
                matches!(result, CegarResult::Verified),
                "{refinement_strategy:?}: state 2 is unreachable, got {result:?}"
            );
        }
        // Connecting state 0 to state 1 makes the same abstract trace real.
        let mc = ModelChecker::new(vec![vec![0.0], vec![0.5], vec![1.0]], vec![(0, 1), (1, 2)]);
        let system = TransitionSystem::new(mc, 0);
        match cegar_verify(&system, &property, &CegarConfig::default()) {
            CegarResult::Violated { trace } => {
                assert_eq!(trace, vec![vec![0.0], vec![0.5], vec![1.0]]);
            }
            other => panic!("Expected Violated, got {other:?}"),
        }
    }
    #[test]
    fn test_cegar_refinement_stalls_when_no_coordinate_separates_the_states() {
        // States 0 and 1 have the same coordinates; only state 1 reaches the
        // violating state 2.  No threshold can tell them apart.
        let mc = ModelChecker::new(vec![vec![0.0], vec![0.0], vec![1.0]], vec![(1, 2)]);
        let system = TransitionSystem::new(mc, 0);
        let property = SafetyProperty::new("x < 1", |s| s[0] < 1.0);
        for refinement_strategy in ALL_STRATEGIES {
            let config = CegarConfig {
                refinement_strategy,
                ..CegarConfig::default()
            };
            match cegar_verify(&system, &property, &config) {
                CegarResult::Unknown { reason } => assert!(reason.contains("stalled"), "{reason}"),
                other => panic!("{refinement_strategy:?}: expected Unknown, got {other:?}"),
            }
        }
    }
    #[test]
    fn test_cegar_result_does_not_depend_on_hash_iteration_order() {
        // Two shortest abstract paths lead to the violating state 3: through
        // state 1 and through state 2.  The search always takes the first one.
        let mc = ModelChecker::new(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            vec![(0, 2), (0, 1), (2, 3), (1, 3)],
        );
        let system = TransitionSystem::new(mc, 0);
        let property = SafetyProperty::new("not both", |s| !(s[0] > 0.5 && s[1] > 0.5));
        let expected = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        for _ in 0..40 {
            match cegar_verify(&system, &property, &CegarConfig::default()) {
                CegarResult::Violated { trace } => assert_eq!(trace, expected),
                other => panic!("Expected Violated, got {other:?}"),
            }
        }
    }
    #[test]
    fn test_cegar_agrees_with_explicit_reachability_on_random_systems() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move |modulus: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % modulus
        };
        let (mut verified, mut violated) = (0usize, 0usize);
        for case in 0..400 {
            let n = 2 + next(11) as usize;
            // The first coordinate is distinct for every state, the other two
            // make states look alike for the initial predicates.
            let states: Vec<Vec<f64>> = (0..n)
                .map(|i| vec![i as f64, next(4) as f64, next(3) as f64])
                .collect();
            let edge_count = n + next(2 * n as u64) as usize;
            let transitions: Vec<(StateIdx, StateIdx)> = (0..edge_count)
                .map(|_| (next(n as u64) as usize, next(n as u64) as usize))
                .collect();
            let initial = next(n as u64) as usize;
            let modulus = 2 + next(4);
            let residue = next(modulus);
            let property = SafetyProperty::new("residue class is bad", move |s| {
                (s[0] as u64) % modulus != residue
            });
            let system = TransitionSystem::new(ModelChecker::new(states, transitions), initial);
            let safe = system
                .model
                .satisfies_invariant(initial, |s| property.check(s));
            let strategy = ALL_STRATEGIES[case % ALL_STRATEGIES.len()];
            let config = CegarConfig {
                max_iterations: n + 2,
                refinement_strategy: strategy,
                ..CegarConfig::default()
            };
            match cegar_verify(&system, &property, &config) {
                CegarResult::Verified => {
                    assert!(
                        safe,
                        "case {case}: Verified, but a violating state is reachable"
                    );
                    verified += 1;
                }
                CegarResult::Violated { trace } => {
                    assert!(!safe, "case {case}: Violated, but the system is safe");
                    assert_real_violating_execution(&system, &property, &trace);
                    violated += 1;
                }
                CegarResult::Unknown { reason } => {
                    panic!("case {case} ({strategy:?}): a finite system must be decided: {reason}")
                }
            }
        }
        assert!(verified >= 20 && violated >= 20, "{verified} / {violated}");
    }
    #[test]
    fn test_symbolic_step_ok() {
        let mut state: SymbolicState = HashMap::new();
        state.insert("x".to_string(), IntervalDomain::new(0.0, 5.0));
        let bounds = IntervalDomain::new(0.0, 10.0);
        let next = symbolic_step(&state, "x", 1.0, &bounds);
        assert!(next.is_some());
    }
    #[test]
    fn test_symbolic_step_out_of_bounds() {
        let mut state: SymbolicState = HashMap::new();
        state.insert("x".to_string(), IntervalDomain::new(9.0, 10.0));
        let bounds = IntervalDomain::new(0.0, 5.0);
        let next = symbolic_step(&state, "x", 3.0, &bounds);
        assert!(next.is_none());
    }
}
