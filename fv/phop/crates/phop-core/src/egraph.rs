//! e-graph canonicalization of a discovered law (opt-in `egraph` feature).
//!
//! phop's default LaTeX rendering ([`crate::distill::canonical_latex`]) lowers a recovered EML tree
//! to the elementary algebra and runs `oxieml`'s rule-based simplifier. That is fast and good, but a
//! single-pass simplifier misses rewrites that only fire after an intermediate non-simplifying step
//! (e.g. `ln(exp(x)) → x`, distributing a product, collapsing a power form). An **e-graph**
//! (equality saturation) explores many rewrite orderings at once and extracts the smallest member of
//! the resulting equivalence class.
//!
//! The e-graph lives in the cool-japan `scirs2-symbolic` crate, whose `LoweredOp` is a *distinct*
//! type from `oxieml`'s (different child pointer, different variant set): `oxieml` carries the special
//! functions (`erf`, `Γ`, `Ei`, …) and lowers `sqrt` to a `Pow`, while `scirs2-symbolic` has native
//! `Sqrt`/`Abs` but no special functions. So a recovered law is first lowered and simplified by
//! `oxieml`, then converted; any subtree `scirs2-symbolic` cannot represent makes the whole
//! conversion bail and we fall back to `oxieml`'s own canonical LaTeX. Plain EML trees only ever
//! lower to `exp`/`ln`/arithmetic, so in practice the conversion always succeeds.

use oxieml::EmlTree;
use scirs2_symbolic::cas::{canonicalize_egraph, SaturationBudget};
use scirs2_symbolic::eml::{to_latex, LoweredOp as SOp};

/// Equality-saturation budget: bounds the e-graph so a pathological input cannot blow up.
const MAX_ITERATIONS: u32 = 30;
/// Node-count ceiling for the e-graph.
const MAX_NODES: u32 = 10_000;

/// Convert an `oxieml` lowered op to `scirs2-symbolic`'s, or `None` if it contains a variant
/// `scirs2-symbolic` lacks (the special functions). `NamedConst` is folded to its numeric value.
fn to_scirs(o: &oxieml::LoweredOp) -> Option<SOp> {
    use oxieml::LoweredOp as O;
    let conv = |a: &oxieml::LoweredOp| to_scirs(a).map(Box::new);
    Some(match o {
        O::Const(c) => SOp::Const(*c),
        O::Var(i) => SOp::Var(*i),
        O::NamedConst(nc) => SOp::Const(nc.value()),
        O::Add(a, b) => SOp::Add(conv(a)?, conv(b)?),
        O::Sub(a, b) => SOp::Sub(conv(a)?, conv(b)?),
        O::Mul(a, b) => SOp::Mul(conv(a)?, conv(b)?),
        O::Div(a, b) => SOp::Div(conv(a)?, conv(b)?),
        O::Pow(a, b) => SOp::Pow(conv(a)?, conv(b)?),
        O::Neg(a) => SOp::Neg(conv(a)?),
        O::Exp(a) => SOp::Exp(conv(a)?),
        O::Ln(a) => SOp::Ln(conv(a)?),
        O::Sin(a) => SOp::Sin(conv(a)?),
        O::Cos(a) => SOp::Cos(conv(a)?),
        O::Tan(a) => SOp::Tan(conv(a)?),
        O::Sinh(a) => SOp::Sinh(conv(a)?),
        O::Cosh(a) => SOp::Cosh(conv(a)?),
        O::Tanh(a) => SOp::Tanh(conv(a)?),
        O::Arcsin(a) => SOp::Arcsin(conv(a)?),
        O::Arccos(a) => SOp::Arccos(conv(a)?),
        O::Arctan(a) => SOp::Arctan(conv(a)?),
        O::Arcsinh(a) => SOp::Arcsinh(conv(a)?),
        O::Arccosh(a) => SOp::Arccosh(conv(a)?),
        O::Arctanh(a) => SOp::Arctanh(conv(a)?),
        // Special functions (Erf, LGamma, Digamma, Trigamma, Ei, Si, Ci) have no scirs2-symbolic
        // counterpart: bail so the caller falls back to oxieml's own LaTeX.
        _ => return None,
    })
}

/// Canonical LaTeX for a tree via `scirs2-symbolic`'s e-graph, falling back to `oxieml`'s
/// rule-based canonical LaTeX when the lowered form contains a special function.
#[must_use]
pub fn canonical_latex_egraph(tree: &EmlTree) -> String {
    let simplified = tree.lower().simplify();
    match to_scirs(&simplified) {
        Some(op) => {
            let budget = SaturationBudget {
                max_iterations: MAX_ITERATIONS,
                max_nodes: MAX_NODES,
            };
            to_latex(canonicalize_egraph(&op, Some(budget)).op())
        }
        None => simplified.to_latex(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxieml::Canonical;

    #[test]
    fn collapses_ln_exp() {
        // ln(exp(x)) is a classic multi-step rewrite: the e-graph must canonicalize it to just `x`,
        // with no leftover transcendental wrapper.
        let tree = Canonical::ln(&Canonical::exp(&EmlTree::var(0)));
        let eg = canonical_latex_egraph(&tree);
        assert!(!eg.is_empty());
        assert!(eg.contains('x'), "expected the bare variable, got: {eg}");
        assert!(!eg.contains("\\ln"), "ln not eliminated: {eg}");
        assert!(
            !eg.contains("exp") && !eg.contains("e^"),
            "exp not eliminated: {eg}"
        );
    }

    #[test]
    fn egraph_is_no_worse_than_oxieml() {
        // For a normal recovered law (exp(x)), the e-graph form is valid and no longer than the
        // plain oxieml canonical LaTeX.
        let tree = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let eg = canonical_latex_egraph(&tree);
        let plain = tree.lower().simplify().to_latex();
        assert!(!eg.is_empty());
        assert!(eg.len() <= plain.len() + 2, "egraph={eg} plain={plain}");
    }
}
