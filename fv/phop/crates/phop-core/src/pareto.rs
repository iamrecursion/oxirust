//! Pareto front over (complexity, MSE).
//!
//! Scientific discovery is **Pareto-first**: rather than returning a single "best" fit, phop
//! returns the accuracy/complexity trade-off front, because a slightly worse but much simpler
//! expression is often the true law (Occam's razor).
//!
//! Beyond the core (complexity, MSE) front, [`ParetoFront::rank_multiobjective`] ranks solutions on
//! **four** objectives — accuracy, complexity, *interpretability* (shallow trees read more easily)
//! and *elegance* (round / named constants) — via `scirs2-optimize`'s NSGA-III machinery
//! (non-dominated sorting + crowding distance), with no evolutionary loop: phop already has the
//! candidates, it just needs them ranked.

use crate::solution::Solution;
use oxieml::EmlNode;
use scirs2_optimize::multiobjective::pareto::{crowding_distance, non_dominated_sort};

/// The non-dominated set of discovered solutions, sorted by complexity ascending.
#[derive(Debug, Clone, Default)]
pub struct ParetoFront {
    /// Non-dominated solutions, ordered by increasing complexity.
    pub solutions: Vec<Solution>,
}

impl ParetoFront {
    /// Build a Pareto front from candidate solutions, discarding dominated ones.
    ///
    /// Near-duplicate solutions (same complexity and near-equal MSE) are collapsed.
    #[must_use]
    pub fn from_candidates(candidates: Vec<Solution>) -> Self {
        let mut front: Vec<Solution> = Vec::new();
        for cand in candidates {
            if !cand.mse.is_finite() {
                continue;
            }
            // Skip if dominated by an existing front member.
            if front.iter().any(|s| s.dominates(&cand)) {
                continue;
            }
            // Remove members the candidate dominates.
            front.retain(|s| !cand.dominates(s));
            // Avoid near-duplicates at equal complexity.
            if front
                .iter()
                .any(|s| s.complexity == cand.complexity && (s.mse - cand.mse).abs() < 1e-12)
            {
                continue;
            }
            front.push(cand);
        }
        front.sort_by(|a, b| {
            a.complexity.cmp(&b.complexity).then(
                a.mse
                    .partial_cmp(&b.mse)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        Self { solutions: front }
    }

    /// The `k` most accurate solutions (lowest MSE first).
    #[must_use]
    pub fn pareto_top(&self, k: usize) -> Vec<&Solution> {
        let mut by_mse: Vec<&Solution> = self.solutions.iter().collect();
        by_mse.sort_by(|a, b| {
            a.mse
                .partial_cmp(&b.mse)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        by_mse.into_iter().take(k).collect()
    }

    /// The single most accurate solution, if any.
    #[must_use]
    pub fn best(&self) -> Option<&Solution> {
        self.solutions.iter().min_by(|a, b| {
            a.mse
                .partial_cmp(&b.mse)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Number of solutions on the front.
    #[must_use]
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Whether the front is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Rank the front's solutions on **four** objectives — accuracy, complexity, interpretability,
    /// elegance — via NSGA-III-style non-dominated sorting plus crowding distance.
    ///
    /// Returns one [`MultiRank`] per solution (same indexing as [`Self::solutions`]), sorted best
    /// first: by non-domination `front` ascending, then by `crowding` distance descending (favouring
    /// diverse, boundary solutions). All four objectives are reconciled to "lower is better" before
    /// sorting (interpretability/elegance are higher-is-better, so they are negated internally).
    #[must_use]
    pub fn rank_multiobjective(&self) -> Vec<MultiRank> {
        if self.solutions.is_empty() {
            return Vec::new();
        }
        let objs_struct: Vec<MultiObjective> = self.solutions.iter().map(objectives).collect();
        // Minimisation vector per solution: [complexity, mse, -interpretability, -elegance].
        let objs: Vec<Vec<f64>> = objs_struct
            .iter()
            .map(|o| vec![o.complexity, o.mse, -o.interpretability, -o.elegance])
            .collect();

        let fronts = non_dominated_sort(&objs);
        let mut ranks: Vec<MultiRank> = Vec::with_capacity(self.solutions.len());
        for (front_rank, front_idx) in fronts.iter().enumerate() {
            let front_objs: Vec<Vec<f64>> = front_idx.iter().map(|&i| objs[i].clone()).collect();
            let cd = crowding_distance(&front_objs);
            for (slot, &i) in front_idx.iter().enumerate() {
                ranks.push(MultiRank {
                    index: i,
                    front: front_rank,
                    crowding: cd.get(slot).copied().unwrap_or(0.0),
                    objectives: objs_struct[i],
                });
            }
        }
        ranks.sort_by(|a, b| {
            a.front.cmp(&b.front).then(
                b.crowding
                    .partial_cmp(&a.crowding)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        ranks
    }
}

/// The four discovery objectives evaluated for a single solution. Accuracy and complexity are
/// "lower is better"; interpretability and elegance are "higher is better".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiObjective {
    /// Structural complexity (EML node count); lower is better.
    pub complexity: f64,
    /// Mean-squared error on the training data; lower is better.
    pub mse: f64,
    /// Interpretability in `(0, 1]`: shallower trees score higher (`1/(1+depth)`).
    pub interpretability: f64,
    /// Elegance in `[0, 1]`: the fraction of constants that are round or named (1.0 if none).
    pub elegance: f64,
}

/// A solution's multi-objective ranking within a [`ParetoFront`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRank {
    /// Index into [`ParetoFront::solutions`].
    pub index: usize,
    /// Non-domination front (0 = Pareto-optimal across all four objectives).
    pub front: usize,
    /// Crowding distance within the solution's front (larger = more isolated / diverse).
    pub crowding: f64,
    /// The four objective values.
    pub objectives: MultiObjective,
}

/// Compute the four objectives for a solution.
#[must_use]
pub fn objectives(sol: &Solution) -> MultiObjective {
    MultiObjective {
        complexity: sol.complexity as f64,
        mse: sol.mse,
        interpretability: 1.0 / (1.0 + depth(&sol.tree.root) as f64),
        elegance: elegance(sol),
    }
}

/// Maximum depth of an EML tree (a leaf has depth 0).
fn depth(node: &EmlNode) -> usize {
    match node {
        EmlNode::One | EmlNode::Var(_) | EmlNode::Const(_) => 0,
        EmlNode::Eml { left, right } => 1 + depth(left).max(depth(right)),
    }
}

/// Fraction of a solution's constants that are "simple" — close to a small integer or recognisable
/// as a named constant (π, e, √2, …). A solution with no free constants is maximally elegant (1.0).
fn elegance(sol: &Solution) -> f64 {
    let mut consts = Vec::new();
    crate::fit::collect_consts(&sol.tree.root, &mut consts);
    if consts.is_empty() {
        return 1.0;
    }
    let simple = consts.iter().filter(|&&c| is_simple_constant(c)).count();
    simple as f64 / consts.len() as f64
}

/// Whether a constant reads as "simple": near a small integer, or snappable to a named constant.
fn is_simple_constant(c: f64) -> bool {
    let near_small_int = (c - c.round()).abs() < 1e-4 && c.round().abs() <= 12.0;
    near_small_int || oxieml::symreg::snap_to_named_const(c).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxieml::{Canonical, EmlTree};

    fn sol(mse: f64, complexity: usize) -> Solution {
        Solution {
            tree: Canonical::exp(&EmlTree::var(0)),
            mse,
            complexity,
        }
    }

    #[test]
    fn ranks_on_four_objectives() {
        use oxieml::EmlTree;
        // s_simple: exp(x) — shallow, zero free constants (max elegance), near-perfect fit.
        let s_simple = Solution::new(EmlTree::eml(&EmlTree::var(0), &EmlTree::one()), 1e-9);
        // s_messy: a deeper tree with two odd constants and a much worse fit — dominated on all four
        // objectives (more complex, less accurate, deeper, inelegant).
        let messy = EmlTree::eml(
            &EmlTree::eml(&EmlTree::var(0), &EmlTree::const_val(0.7234)),
            &EmlTree::const_val(0.1119),
        );
        let s_messy = Solution::new(messy, 0.5);

        let front = ParetoFront {
            solutions: vec![s_simple, s_messy],
        };
        let ranks = front.rank_multiobjective();
        assert_eq!(ranks.len(), 2, "one rank per solution");

        // Ranking on four objectives: the simple/accurate/elegant solution is Pareto-optimal and
        // sorted first; the dominated one lands in a later front.
        assert_eq!(ranks[0].front, 0);
        assert_eq!(ranks[0].index, 0);
        let messy_rank = ranks
            .iter()
            .find(|r| r.index == 1)
            .expect("messy solution ranked");
        assert!(
            messy_rank.front >= 1,
            "dominated solution should be a later front"
        );

        // Objective values: exp(x) has no free constants → elegance 1.0; the messy tree's two odd
        // constants are neither round nor named → elegance 0.0.
        assert!((ranks[0].objectives.elegance - 1.0).abs() < 1e-12);
        assert!(messy_rank.objectives.elegance < 1e-12);
        // Interpretability is in (0, 1] and the shallower tree scores higher.
        assert!(ranks[0].objectives.interpretability > messy_rank.objectives.interpretability);
    }

    #[test]
    fn keeps_only_non_dominated() {
        let cands = vec![
            sol(0.5, 1), // non-dominated (simplest)
            sol(0.1, 5), // non-dominated (most accurate)
            sol(0.6, 6), // dominated by both
            sol(0.3, 3), // non-dominated (middle)
        ];
        let front = ParetoFront::from_candidates(cands);
        assert_eq!(front.len(), 3);
        assert!((front.best().unwrap().mse - 0.1).abs() < 1e-12);
        // Sorted by complexity ascending.
        assert!(front.solutions[0].complexity <= front.solutions[1].complexity);
    }
}
