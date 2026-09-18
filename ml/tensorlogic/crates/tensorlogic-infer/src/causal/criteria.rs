//! Backdoor / frontdoor criteria and do-operator graph mutilation.
//!
//! Pure graph-theoretic identification procedures:
//! - [`backdoor_criterion`], [`find_backdoor_adjustment`]
//! - [`frontdoor_criterion`]
//! - [`do_intervention`] (graph mutilation)

use std::collections::{HashSet, VecDeque};

use super::data::{BackdoorAdjustment, Intervention};
use super::error::CausalError;
use super::graph::CausalGraph;

// ---------------------------------------------------------------------------
// Backdoor criterion
// ---------------------------------------------------------------------------

/// Check whether `adjustment_set` satisfies the backdoor criterion for
/// the causal effect of `treatment` on `outcome` in `graph`.
///
/// The backdoor criterion holds when:
/// 1. No node in `adjustment_set` is a descendant of `treatment`.
/// 2. `adjustment_set` blocks all backdoor paths from `treatment` to `outcome`
///    (i.e. paths that enter `treatment` through one of its parents).
pub fn backdoor_criterion(
    graph: &CausalGraph,
    treatment: &str,
    outcome: &str,
    adjustment_set: &[&str],
) -> bool {
    let treatment_idx = match graph.node_index(treatment) {
        Some(i) => i,
        None => return false,
    };
    if graph.node_index(outcome).is_none() {
        return false;
    }

    // Condition 1: no descendant of treatment is in adjustment_set
    let treatment_desc = graph.descendants_of(treatment);
    let treatment_desc_set: HashSet<String> = treatment_desc.into_iter().collect();
    for &z in adjustment_set {
        if treatment_desc_set.contains(z) {
            return false;
        }
    }

    // Condition 2: adjustment_set blocks all backdoor paths
    let adj_idx_set: HashSet<usize> = adjustment_set
        .iter()
        .filter_map(|&z| graph.node_index(z))
        .collect();

    let outcome_idx = graph.node_index(outcome).unwrap_or(usize::MAX);

    // Check: is there still an unblocked backdoor path from treatment to outcome?
    !graph.has_unblocked_backdoor_path(treatment_idx, outcome_idx, &adj_idx_set)
}

/// Find a minimal backdoor adjustment set for the causal effect of `treatment` on `outcome`.
///
/// Strategy: start with the direct parents of `treatment`, then greedily add more
/// nodes if necessary.  Returns an error if no path from treatment to outcome exists.
pub fn find_backdoor_adjustment(
    graph: &CausalGraph,
    treatment: &str,
    outcome: &str,
) -> Result<BackdoorAdjustment, CausalError> {
    if graph.node_index(treatment).is_none() {
        return Err(CausalError::NodeNotFound(treatment.to_string()));
    }
    if graph.node_index(outcome).is_none() {
        return Err(CausalError::NodeNotFound(outcome.to_string()));
    }

    // Try the empty set first
    if backdoor_criterion(graph, treatment, outcome, &[]) {
        return Ok(BackdoorAdjustment {
            adjustment_set: vec![],
            valid: true,
        });
    }

    // Try parents of treatment
    let parents = graph.parents_of(treatment);
    let parent_refs: Vec<&str> = parents.iter().map(|s| s.as_str()).collect();
    if backdoor_criterion(graph, treatment, outcome, &parent_refs) {
        return Ok(BackdoorAdjustment {
            adjustment_set: parents,
            valid: true,
        });
    }

    // Greedy expansion: add ancestors of treatment one by one
    let ancestors = graph.ancestors_of(treatment);
    let treatment_desc: HashSet<String> = graph.descendants_of(treatment).into_iter().collect();

    let mut candidate: Vec<String> = parents;
    for anc in &ancestors {
        if !treatment_desc.contains(anc) && !candidate.contains(anc) {
            candidate.push(anc.clone());
            let refs: Vec<&str> = candidate.iter().map(|s| s.as_str()).collect();
            if backdoor_criterion(graph, treatment, outcome, &refs) {
                return Ok(BackdoorAdjustment {
                    adjustment_set: candidate,
                    valid: true,
                });
            }
        }
    }

    // Return what we have even if not valid (caller can check `valid` flag)
    let refs: Vec<&str> = candidate.iter().map(|s| s.as_str()).collect();
    let valid = backdoor_criterion(graph, treatment, outcome, &refs);
    Ok(BackdoorAdjustment {
        adjustment_set: candidate,
        valid,
    })
}

// ---------------------------------------------------------------------------
// Frontdoor criterion
// ---------------------------------------------------------------------------

/// Check whether `mediator_set` satisfies the frontdoor criterion for
/// the causal effect of `treatment` on `outcome` in `graph`.
///
/// The frontdoor criterion holds when:
/// 1. All directed paths from `treatment` to `outcome` are intercepted by `mediator_set`.
/// 2. There is no unblocked backdoor path from `treatment` to any node in `mediator_set`.
/// 3. All backdoor paths from mediator nodes to `outcome` are blocked by `treatment`.
pub fn frontdoor_criterion(
    graph: &CausalGraph,
    treatment: &str,
    outcome: &str,
    mediator_set: &[&str],
) -> bool {
    if graph.node_index(treatment).is_none() || graph.node_index(outcome).is_none() {
        return false;
    }
    if mediator_set.is_empty() {
        return false;
    }

    // Condition 1: every directed path from treatment to outcome passes through mediator_set
    // Check via: after removing mediator_set from graph, no directed path from treatment→outcome
    // We simulate this by checking if mediator_set blocks all directed paths.
    let mediator_idxs: HashSet<usize> = mediator_set
        .iter()
        .filter_map(|&m| graph.node_index(m))
        .collect();

    // BFS from treatment to outcome excluding mediator nodes as intermediate nodes
    let treatment_idx = match graph.node_index(treatment) {
        Some(i) => i,
        None => return false,
    };
    let outcome_idx = match graph.node_index(outcome) {
        Some(i) => i,
        None => return false,
    };

    // Check if there is a directed path from treatment to outcome that bypasses all mediators
    let bypasses_mediators = {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(treatment_idx);
        let mut found = false;
        while let Some(cur) = queue.pop_front() {
            if cur == outcome_idx {
                found = true;
                break;
            }
            if !visited.insert(cur) {
                continue;
            }
            for &(p, c) in &graph.edges {
                if p == cur
                    && !visited.contains(&c)
                    && (c == outcome_idx || !mediator_idxs.contains(&c))
                {
                    queue.push_back(c);
                }
            }
        }
        found
    };
    if bypasses_mediators {
        return false;
    }

    // Condition 2: no unblocked backdoor path from treatment to any mediator
    // (i.e. mediators are "cleanly" reached from treatment)
    let treatment_set: HashSet<usize> = std::iter::once(treatment_idx).collect();
    for &m in mediator_set {
        let m_idx = match graph.node_index(m) {
            Some(i) => i,
            None => return false,
        };
        // Check: is there an unblocked backdoor path from treatment to m?
        // Using empty adjustment set — if any backdoor path is open, condition fails.
        if graph.has_unblocked_backdoor_path(treatment_idx, m_idx, &HashSet::new()) {
            return false;
        }
        // Alternative: is treatment d-separating all backdoor paths to m from confounders?
        // (the full criterion also checks that all backdoor paths from m to outcome
        //  are blocked by {treatment})
        let _ = treatment_set.len(); // suppress warning
    }

    // Condition 3: all backdoor paths from each mediator to outcome are blocked by {treatment}
    let treatment_as_adj: HashSet<usize> = std::iter::once(treatment_idx).collect();
    for &m in mediator_set {
        let m_idx = match graph.node_index(m) {
            Some(i) => i,
            None => return false,
        };
        if graph.has_unblocked_backdoor_path(m_idx, outcome_idx, &treatment_as_adj) {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// do-intervention (graph mutilation)
// ---------------------------------------------------------------------------

/// Apply a do-intervention by mutilating the causal graph.
///
/// Removes all incoming edges to `intervention.variable`, producing a new graph
/// where the variable is set by external action rather than its natural causes.
/// Outgoing edges (causal effects of the variable) are preserved.
pub fn do_intervention(graph: &CausalGraph, intervention: &Intervention) -> CausalGraph {
    let var_idx = graph.node_index(&intervention.variable);
    let new_edges: Vec<(usize, usize)> = match var_idx {
        None => graph.edges.clone(),
        Some(idx) => graph
            .edges
            .iter()
            .filter(|&&(_, c)| c != idx)
            .cloned()
            .collect(),
    };
    CausalGraph {
        nodes: graph.nodes.clone(),
        edges: new_edges,
    }
}
