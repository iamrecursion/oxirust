// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous label propagation (Raghavan, Albert, Kumara, 2007) on the
//! weighted graph of the community kernels.
//!
//! Every node starts with a label of its own.  The nodes are then visited in
//! ascending id order, and a node takes the label that carries the largest
//! total edge weight among its neighbours (at the time it is visited, so labels
//! spread within a sweep); among labels of equal weight the smallest wins.  The
//! sweeps stop when a whole sweep changes nothing or after `max_iter` sweeps.
//!
//! Visiting order and tie-breaking are fixed, so the result is deterministic.
//! Labels are the ranks of the node ids, so "smallest label" means "smallest
//! id" and the arrays below can be indexed by label.

use super::WeightedGraph;

/// Total weights closer than this (relative to the larger one) count as tied.
/// It absorbs the rounding error of summing weights in different orders.
const TIE_TOLERANCE: f64 = 1e-12;

/// Label of every node of `graph` after at most `max_iter` sweeps.
///
/// `ids[i]` is the id of node `i`; it only determines the visiting order and
/// which label wins a tie.  Labels are numbers below `graph.len()`, not
/// necessarily dense.
pub(super) fn propagate(ids: &[usize], graph: &WeightedGraph, max_iter: usize) -> Vec<usize> {
    let n = graph.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&node| ids[node]);
    let mut label = vec![0_usize; n];
    for (rank, &node) in order.iter().enumerate() {
        label[node] = rank;
    }
    // Total weight from the visited node to every label among its neighbours;
    // `stamp[l]` records the visit that last wrote `weight_of[l]`.
    let mut weight_of = vec![0.0_f64; n];
    let mut stamp = vec![0_usize; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut visit = 0_usize;
    for _ in 0..max_iter {
        let mut changed = false;
        for &node in &order {
            let neighbours = &graph.adj[node];
            if neighbours.is_empty() {
                continue;
            }
            visit += 1;
            touched.clear();
            for &(other, weight) in neighbours {
                let candidate = label[other];
                if stamp[candidate] != visit {
                    stamp[candidate] = visit;
                    weight_of[candidate] = 0.0;
                    touched.push(candidate);
                }
                weight_of[candidate] += weight;
            }
            let mut best: Option<(usize, f64)> = None;
            for &candidate in &touched {
                let weight = weight_of[candidate];
                best = Some(match best {
                    None => (candidate, weight),
                    Some((best_label, best_weight)) => {
                        let tolerance = TIE_TOLERANCE * best_weight.max(weight);
                        let heavier = weight > best_weight + tolerance;
                        let tied_and_smaller =
                            (weight - best_weight).abs() <= tolerance && candidate < best_label;
                        if heavier || tied_and_smaller {
                            (candidate, weight)
                        } else {
                            (best_label, best_weight)
                        }
                    }
                });
            }
            if let Some((winner, _)) = best
                && winner != label[node]
            {
                label[node] = winner;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    label
}

#[cfg(test)]
mod tests;
