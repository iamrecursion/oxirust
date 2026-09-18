//! Layer D — symbolic distillation of a discovered EML tree.
//!
//! A raw EML tree is a composition of a single primitive (`eml(a, b) = exp(a) − ln(b)`). To be
//! useful as a *discovered law* it must be turned into a recognizable closed form and emitted in
//! the formats scientists actually use. This module is the single entry point for that:
//!
//! 1. **Canonicalize** by lowering the EML tree to the general elementary-function algebra and
//!    running `oxieml`'s simplifier (`tree.lower().simplify()`), which folds `ln(1) = 0`,
//!    `exp(ln(x)) = x`, constant arithmetic, etc. — so `eml(x, 1)` distills to `e^{x}`.
//! 2. **Render** the canonical form to LaTeX and a pretty math string, and generate executable
//!    code in Rust, NumPy, and SymPy directly from the AST.
//!
//! Named-constant *snapping* (π, e, √2, …) happens earlier, during fitting ([`crate::polish`]),
//! so the constants reaching distillation are already recognizable where possible.

use crate::codegen;
use oxieml::EmlTree;

/// A discovered expression rendered into every output format phop supports.
#[derive(Debug, Clone)]
pub struct Distilled {
    /// Canonical LaTeX (after lowering + algebraic simplification).
    pub latex: String,
    /// Canonical pretty math string (e.g. `exp(x0) - ln(2)`), simplified.
    pub pretty: String,
    /// A standalone Rust function `f(x0, …) -> f64`.
    pub rust: String,
    /// A NumPy-compatible Python lambda.
    pub numpy: String,
    /// A SymPy expression string (with a leading `symbols(...)` hint comment).
    pub sympy: String,
    /// Structural complexity (EML node count) of the original tree.
    pub complexity: usize,
}

/// Canonical LaTeX for a tree: lower to the elementary algebra and simplify before rendering.
#[must_use]
pub fn canonical_latex(tree: &EmlTree) -> String {
    tree.lower().simplify().to_latex()
}

/// Canonical pretty math string for a tree (simplified rich-AST display).
#[must_use]
pub fn canonical_pretty(tree: &EmlTree) -> String {
    format!("{}", tree.lower().simplify())
}

/// Distill a tree into all supported output formats.
#[must_use]
pub fn distill(tree: &EmlTree) -> Distilled {
    Distilled {
        latex: canonical_latex(tree),
        pretty: canonical_pretty(tree),
        rust: codegen::rust_code(tree),
        numpy: codegen::numpy_code(tree),
        sympy: codegen::sympy_code(tree),
        complexity: tree.size(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_distills_to_clean_forms() {
        // eml(x, 1) = exp(x) - ln(1) = exp(x); canonicalization should drop the ln(1) term.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let d = distill(&tree);
        // LaTeX simplifies to e^{x_0} with no leftover "ln" of one.
        assert!(
            d.latex.contains("e^") || d.latex.contains("exp"),
            "latex: {}",
            d.latex
        );
        assert!(
            !d.latex.contains("\\ln\\left(1"),
            "ln(1) not simplified: {}",
            d.latex
        );
        // Every code target is populated.
        assert!(d.rust.contains("fn f("));
        assert!(d.numpy.contains("np.exp"));
        assert!(d.sympy.contains("exp("));
        assert_eq!(d.complexity, tree.size());
    }

    #[test]
    fn canonical_latex_matches_solution_latex() {
        // distill's canonical LaTeX is the same path Solution::latex uses.
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        assert_eq!(canonical_latex(&tree), tree.lower().simplify().to_latex());
    }
}
