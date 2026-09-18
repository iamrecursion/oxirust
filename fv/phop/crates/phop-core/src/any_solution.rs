//! A heterogeneous discovered law — either an **EML-tree** law from the differentiable core engine
//! or a **rich-leaf** affine/log-linear law from [`crate::affine`].
//!
//! The two engines deliberately use different representations (`oxieml::EmlTree` vs the affine
//! engine's own `ANode`, because `a·x+b` / `Σaᵢ ln xᵢ` with negative coefficients is not a *guarded
//! real* EML form). [`AnySolution`] unifies them at the accuracy/complexity layer so both can live on
//! a single Pareto front and render through one path — which is what lets `discover_auto_all` (and the
//! CLI `auto` method) **reach the affine engine by default**. The affine engine is the one that
//! recovers product / power / ratio laws the shallow EML core can't assemble, so the default search
//! must include it.

use crate::affine::AffineSolution;
use crate::error::Result;
use crate::solution::Solution;
use scirs2_core::ndarray::{Array1, Array2};

/// A discovered law from either engine, exposing a common accuracy/complexity interface.
#[derive(Clone, Debug)]
pub enum AnySolution {
    /// A law from the EML-tree core (`enumerate` / Gumbel / gated / `oxieml`-symreg). Carries the
    /// full downstream capability surface (analyze / certified-root / proofs / codegen) via the
    /// inner [`Solution`].
    Eml(Solution),
    /// A law from the rich-leaf affine engine (`Linear` / `LogLinear` leaves over guarded `eml`).
    Affine(AffineSolution),
}

impl AnySolution {
    /// Structural complexity (lower is simpler): EML node count for [`Self::Eml`], active-parameter
    /// count for [`Self::Affine`]. Both measure "number of structural pieces" — commensurable enough
    /// to share one Pareto axis.
    #[must_use]
    pub fn complexity(&self) -> usize {
        match self {
            Self::Eml(s) => s.complexity,
            Self::Affine(s) => s.nodes,
        }
    }

    /// Mean-squared error on the training data (lower is better).
    #[must_use]
    pub fn mse(&self) -> f64 {
        match self {
            Self::Eml(s) => s.mse,
            Self::Affine(s) => s.mse,
        }
    }

    /// A short identifier of the producing engine (`"eml"` / `"affine"`), for display.
    #[must_use]
    pub fn source(&self) -> &'static str {
        match self {
            Self::Eml(_) => "eml",
            Self::Affine(_) => "affine",
        }
    }

    /// Whether this is a clean **symbolic** recovery. An EML law is exact in its own algebra, so it
    /// reports `true`; an affine law reports `true` only when its exponents/slopes snapped to small
    /// rationals / named constants while retaining R² ≥ 0.999 (see [`AffineSolution::symbolic`]).
    #[must_use]
    pub fn is_symbolic(&self) -> bool {
        match self {
            Self::Eml(_) => true,
            Self::Affine(s) => s.symbolic,
        }
    }

    /// A human-readable expression.
    #[must_use]
    pub fn expr(&self) -> String {
        match self {
            Self::Eml(s) => s.pretty(),
            Self::Affine(s) => s.expr.clone(),
        }
    }

    /// A LaTeX rendering of the law.
    #[must_use]
    pub fn latex(&self) -> String {
        match self {
            Self::Eml(s) => s.latex(),
            Self::Affine(s) => s.latex(),
        }
    }

    /// Evaluate the law on new data `x` (`[n_rows, n_vars]`).
    ///
    /// # Errors
    /// Returns [`crate::PhopError`] if the EML forward fails (the affine forward is total).
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        match self {
            Self::Eml(s) => s.predict(x),
            Self::Affine(s) => Ok(s.predict(x)),
        }
    }

    /// The underlying EML [`Solution`] if this came from the core engine — the handle for the
    /// analyze / certified-root / proof / codegen capabilities the affine engine does not provide.
    #[must_use]
    pub fn as_eml(&self) -> Option<&Solution> {
        match self {
            Self::Eml(s) => Some(s),
            Self::Affine(_) => None,
        }
    }

    /// Weak Pareto domination over (complexity, MSE): no worse in either, strictly better in one.
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        let (c0, m0) = (self.complexity(), self.mse());
        let (c1, m1) = (other.complexity(), other.mse());
        c0 <= c1 && m0 <= m1 && (c0 < c1 || m0 < m1)
    }
}

/// Pareto-filter a heterogeneous candidate set over (complexity, MSE) — discarding dominated and
/// near-duplicate members — then sort by MSE ascending (most accurate first). Mirrors
/// [`crate::pareto::ParetoFront::from_candidates`] but over [`AnySolution`].
#[must_use]
pub fn merge_pareto(candidates: Vec<AnySolution>) -> Vec<AnySolution> {
    let mut front: Vec<AnySolution> = Vec::new();
    for cand in candidates {
        if !cand.mse().is_finite() {
            continue;
        }
        if front.iter().any(|s| s.dominates(&cand)) {
            continue;
        }
        front.retain(|s| !cand.dominates(s));
        if front
            .iter()
            .any(|s| s.complexity() == cand.complexity() && (s.mse() - cand.mse()).abs() < 1e-12)
        {
            continue;
        }
        front.push(cand);
    }
    front.sort_by(|a, b| {
        a.mse()
            .partial_cmp(&b.mse())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    front
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::discover_affine_pareto;
    use crate::solution::Solution;
    use oxieml::EmlTree;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn domination_uses_complexity_and_mse() {
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let simple_accurate = AnySolution::Eml(Solution {
            tree: tree.clone(),
            mse: 0.1,
            complexity: 3,
        });
        let complex_worse = AnySolution::Eml(Solution {
            tree,
            mse: 0.2,
            complexity: 5,
        });
        assert!(simple_accurate.dominates(&complex_worse));
        assert!(!complex_worse.dominates(&simple_accurate));
    }

    #[test]
    fn merge_keeps_non_dominated_and_sorts_by_mse() {
        let t = oxieml::Canonical::exp(&EmlTree::var(0));
        let a = AnySolution::Eml(Solution {
            tree: t.clone(),
            mse: 0.30,
            complexity: 2,
        });
        let b = AnySolution::Eml(Solution {
            tree: t.clone(),
            mse: 0.10,
            complexity: 4,
        });
        // Dominated by `a` (worse complexity, worse mse) — must be dropped.
        let dominated = AnySolution::Eml(Solution {
            tree: t,
            mse: 0.40,
            complexity: 6,
        });
        let front = merge_pareto(vec![a, b, dominated]);
        assert_eq!(front.len(), 2, "the dominated member must be removed");
        // Sorted by MSE ascending.
        assert!(front[0].mse() <= front[1].mse());
        assert!((front[0].mse() - 0.10).abs() < 1e-12);
    }

    #[test]
    fn merge_includes_affine_members() {
        // y = x0^2 * x1 — a product/power law the affine (LogLinear) engine recovers but the shallow
        // EML core does not. After merging, at least one Affine member must be present and accurate.
        let n = 40usize;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = 1.0 + 0.05 * i as f64;
            let x1 = 0.5 + 0.03 * i as f64;
            x[[i, 0]] = x0;
            x[[i, 1]] = x1;
            y[i] = x0 * x0 * x1;
        }
        let affine = discover_affine_pareto(&x, &y, 2, 500);
        let merged = merge_pareto(affine.into_iter().map(AnySolution::Affine).collect());
        assert!(!merged.is_empty(), "affine engine should recover x0^2*x1");
        let best = &merged[0];
        assert_eq!(best.source(), "affine");
        assert!(best.mse() < 1e-3, "best affine mse = {}", best.mse());
    }
}
