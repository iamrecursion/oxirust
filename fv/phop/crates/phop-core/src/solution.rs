//! A discovered candidate expression and its quality metrics.

use crate::codegen;
use crate::error::Result;
use oxieml::EmlTree;
use scirs2_core::ndarray::{Array1, Array2};

/// A single discovered expression together with its accuracy and complexity.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The EML tree (with fitted constants).
    pub tree: EmlTree,
    /// Mean-squared error on the training data.
    pub mse: f64,
    /// Structural complexity (EML node count).
    pub complexity: usize,
}

impl Solution {
    /// Build a solution from a tree and its MSE; complexity is the node count.
    #[must_use]
    pub fn new(tree: EmlTree, mse: f64) -> Self {
        let complexity = tree.size();
        Self {
            tree,
            mse,
            complexity,
        }
    }

    /// Render the expression as a LaTeX math string (via oxieml lowering + simplification).
    #[must_use]
    pub fn latex(&self) -> String {
        self.tree.lower().simplify().to_latex()
    }

    /// A human-readable EML form, e.g. `eml(x0, 1)`.
    #[must_use]
    pub fn pretty(&self) -> String {
        format!("{}", self.tree)
    }

    /// Render the expression as a **stronger** canonical LaTeX via `scirs2-symbolic`'s e-graph
    /// (equality saturation), which collapses forms a single-pass simplifier misses (e.g.
    /// `ln(exp(x)) → x`). Falls back to [`Self::latex`] for laws containing special functions.
    /// Requires the `egraph` feature.
    #[cfg(feature = "egraph")]
    #[must_use]
    pub fn latex_egraph(&self) -> String {
        crate::egraph::canonical_latex_egraph(&self.tree)
    }

    /// Generate a standalone Rust function for the expression.
    #[must_use]
    pub fn rust_code(&self) -> String {
        codegen::rust_code(&self.tree)
    }

    /// Generate a NumPy-compatible Python lambda for the expression.
    #[must_use]
    pub fn numpy_code(&self) -> String {
        codegen::numpy_code(&self.tree)
    }

    /// Generate a SymPy expression string for the expression.
    #[must_use]
    pub fn sympy_code(&self) -> String {
        codegen::sympy_code(&self.tree)
    }

    /// Distill the expression into every supported output format at once
    /// (canonical LaTeX/pretty plus Rust/NumPy/SymPy code).
    #[must_use]
    pub fn distill(&self) -> crate::distill::Distilled {
        crate::distill::distill(&self.tree)
    }

    /// Evaluate the discovered expression on new data `x` (`[n_rows, n_vars]`), returning the
    /// predictions `[n_rows]` through phop's guarded forward.
    ///
    /// # Errors
    /// Returns [`crate::PhopError`] if evaluation fails or produces non-finite values.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        crate::forest::eval_tree(&self.tree, x)
    }

    /// Serialize the discovered law (its EML tree) to a portable JSON string — the model artifact
    /// `phop predict` consumes. Round-trips with [`Self::from_model_json`].
    ///
    /// # Errors
    /// Returns [`crate::PhopError::Symbolic`] if serialization fails.
    pub fn to_model_json(&self) -> Result<String> {
        serde_json::to_string(&self.tree)
            .map_err(|e| crate::error::PhopError::Symbolic(format!("model serialize: {e}")))
    }

    /// Reconstruct a law from [`Self::to_model_json`] output. The MSE is unknown until the law is
    /// re-evaluated (left as `NaN`); complexity is recomputed from the tree.
    ///
    /// # Errors
    /// Returns [`crate::PhopError::Parse`] if the JSON is not a valid serialized EML tree.
    pub fn from_model_json(s: &str) -> Result<Self> {
        let tree: EmlTree = serde_json::from_str(s)
            .map_err(|e| crate::error::PhopError::Parse(format!("model deserialize: {e}")))?;
        Ok(Self::new(tree, f64::NAN))
    }

    /// Symbolically analyze the discovered law via the oxieml CAS: derivative, antiderivative,
    /// Maclaurin series, and the `+∞` limit, all w.r.t. variable `wrt` (see [`crate::analyze()`]).
    #[must_use]
    pub fn analyze(&self, wrt: usize, series_order: usize) -> crate::analyze::Analysis {
        crate::analyze::analyze(&self.tree, wrt, series_order)
    }

    /// Find a **certified** root of the law in `[lo, hi]` along variable `wrt` (interval
    /// Newton/Krawczyk verification). `others` fixes the remaining variables (`&[]` for 1-D).
    ///
    /// # Errors
    /// Returns [`crate::PhopError`] if the verifier errors.
    pub fn certified_root(
        &self,
        wrt: usize,
        others: &[f64],
        lo: f64,
        hi: f64,
    ) -> Result<oxieml::RootCertificate> {
        crate::analyze::certified_root(&self.tree, wrt, others, lo, hi)
    }

    /// A **guaranteed** interval enclosure of the law's range over an axis-aligned `domain`
    /// (`[lo, hi]` per variable): `f(x) ∈ [returned.0, returned.1]` for all `x` in the box.
    #[must_use]
    pub fn certified_range(&self, domain: &[(f64, f64)]) -> (f64, f64) {
        crate::analyze::certified_range(&self.tree, domain)
    }

    /// Generate a standalone, **canonical** Rust function for the law (lowered + simplified over the
    /// full elementary algebra via `oxieml`), suitable for embedding/deployment. This differs from
    /// [`Self::rust_code`], which emits the raw `eml`-form; here the recovered law is simplified to
    /// ordinary `exp/ln/+/-/…` first.
    #[must_use]
    pub fn compile_rust(&self, fn_name: &str) -> String {
        oxieml::compile::compile_to_rust(&self.tree, fn_name)
    }

    /// **Prove** the law has no root over `bounds` (one `(lo, hi)` per variable) using the OxiZ
    /// SMT backend. [`Verdict::Proven`](crate::verify::Verdict::Proven) is sound — it certifies
    /// `f(x) ≠ 0` over the whole box. Requires the `smt` feature.
    #[cfg(feature = "smt")]
    #[must_use]
    pub fn prove_no_root(&self, bounds: &[(f64, f64)]) -> crate::verify::Verdict {
        crate::verify::prove_no_root(&self.tree, bounds)
    }

    /// **Prove** the lower bound `f(x) ≥ c` over `bounds` via the OxiZ SMT backend. Requires `smt`.
    #[cfg(feature = "smt")]
    #[must_use]
    pub fn prove_lower_bound(&self, c: f64, bounds: &[(f64, f64)]) -> crate::verify::Verdict {
        crate::verify::prove_lower_bound(&self.tree, c, bounds)
    }

    /// **Prove** the upper bound `f(x) ≤ c` over `bounds` via the OxiZ SMT backend. Requires `smt`.
    #[cfg(feature = "smt")]
    #[must_use]
    pub fn prove_upper_bound(&self, c: f64, bounds: &[(f64, f64)]) -> crate::verify::Verdict {
        crate::verify::prove_upper_bound(&self.tree, c, bounds)
    }

    /// **Prove** this law is equal to `other` over `bounds` via the OxiZ SMT backend. Requires `smt`.
    #[cfg(feature = "smt")]
    #[must_use]
    pub fn prove_equivalent(
        &self,
        other: &Solution,
        bounds: &[(f64, f64)],
    ) -> crate::verify::Verdict {
        crate::verify::prove_equivalent(&self.tree, &other.tree, bounds)
    }

    /// Map this discovered law to a **TensorLogic** expression (`tensorlogic_ir::TLExpr`) — the
    /// neuro-symbolic bridge. The law is lowered + simplified, then translated by oxieml's
    /// `tensorlogic` IR, so a recovered equation can enter a differentiable logic program. Requires
    /// the `tensorlogic` feature.
    #[cfg(feature = "tensorlogic")]
    #[must_use]
    pub fn to_tlexpr(&self) -> tensorlogic_ir::TLExpr {
        oxieml::tensorlogic::to_tlexpr(&self.tree.lower().simplify())
    }

    /// Map this law to a **weighted** TensorLogic rule (`weight :: rule`), the form a neuro-symbolic
    /// engine ingests as a soft constraint. Requires the `tensorlogic` feature.
    #[cfg(feature = "tensorlogic")]
    #[must_use]
    pub fn to_tl_weighted_rule(&self, weight: f64) -> tensorlogic_ir::TLExpr {
        tensorlogic_ir::TLExpr::WeightedRule {
            weight,
            rule: Box::new(self.to_tlexpr()),
        }
    }

    /// Map this law to a weighted TensorLogic **equation** `target_var = f(…)` (the predicted target
    /// on the left, the discovered expression on the right). Requires the `tensorlogic` feature.
    #[cfg(feature = "tensorlogic")]
    #[must_use]
    pub fn to_tl_weighted_equation(&self, target_var: &str, weight: f64) -> tensorlogic_ir::TLExpr {
        let lhs = tensorlogic_ir::TLExpr::Pred {
            name: target_var.to_string(),
            args: vec![tensorlogic_ir::Term::var(target_var)],
        };
        let eq = tensorlogic_ir::TLExpr::Eq(Box::new(lhs), Box::new(self.to_tlexpr()));
        tensorlogic_ir::TLExpr::WeightedRule {
            weight,
            rule: Box::new(eq),
        }
    }

    /// Does this solution (weakly) Pareto-dominate `other`?
    ///
    /// Domination minimizes both complexity and MSE, with strict improvement in at least one.
    #[must_use]
    pub fn dominates(&self, other: &Solution) -> bool {
        self.complexity <= other.complexity
            && self.mse <= other.mse
            && (self.complexity < other.complexity || self.mse < other.mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latex_renders() {
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let sol = Solution::new(tree, 0.0);
        let tex = sol.latex();
        assert!(!tex.is_empty());
    }

    #[test]
    fn predict_matches_forward_eval() {
        use scirs2_core::ndarray::Array2;
        // eml(x0, 1) = exp(x0). predict should reproduce exp on fresh inputs.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let sol = Solution::new(tree, 0.0);
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let p = sol.predict(&x).unwrap();
        for (i, &xi) in [0.0_f64, 1.0, 2.0].iter().enumerate() {
            assert!(
                (p[i] - xi.exp()).abs() < 1e-9,
                "row {i}: {} vs {}",
                p[i],
                xi.exp()
            );
        }
    }

    #[test]
    fn model_json_round_trips() {
        use scirs2_core::ndarray::Array2;
        // eml(x0, 1) = exp(x0). Serialize → deserialize → the law must predict identically.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let sol = Solution::new(tree, 0.0);
        let json = sol.to_model_json().expect("serialize");
        let back = Solution::from_model_json(&json).expect("deserialize");
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).unwrap();
        let a = sol.predict(&x).unwrap();
        let b = back.predict(&x).unwrap();
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert!(
                (pa - pb).abs() < 1e-12,
                "round-trip changed prediction: {pa} vs {pb}"
            );
        }
        assert!(Solution::from_model_json("not json").is_err());
    }

    #[test]
    fn compile_rust_emits_a_function() {
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let code = Solution::new(tree, 0.0).compile_rust("kepler");
        assert!(
            code.contains("fn kepler"),
            "expected a named fn, got: {code}"
        );
    }

    #[cfg(feature = "tensorlogic")]
    #[test]
    fn maps_to_weighted_logic_rule() {
        // A discovered law becomes a weighted TensorLogic rule carrying the weight verbatim, and the
        // equation form wraps an `Eq` under a `WeightedRule`.
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let sol = Solution::new(tree, 0.0);

        let weight = match sol.to_tl_weighted_rule(0.7) {
            tensorlogic_ir::TLExpr::WeightedRule { weight, .. } => weight,
            _ => f64::NAN,
        };
        assert!(
            (weight - 0.7).abs() < 1e-12,
            "expected a WeightedRule carrying 0.7, got {weight}"
        );

        let eq = sol.to_tl_weighted_equation("y", 1.0);
        assert!(matches!(eq, tensorlogic_ir::TLExpr::WeightedRule { .. }));
    }

    #[test]
    fn domination_is_correct() {
        let t = oxieml::Canonical::exp(&EmlTree::var(0));
        let a = Solution {
            tree: t.clone(),
            mse: 0.1,
            complexity: 3,
        };
        let b = Solution {
            tree: t,
            mse: 0.2,
            complexity: 5,
        };
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }
}
