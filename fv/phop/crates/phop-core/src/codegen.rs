//! Code generation for discovered EML expressions (Rust, Python/NumPy).
//!
//! LaTeX rendering is delegated to `oxieml` (`tree.lower().simplify().to_latex()`); this
//! module emits executable source in other targets directly from the EML AST.

use oxieml::{EmlNode, EmlTree};

fn rust_expr(node: &EmlNode) -> String {
    match node {
        EmlNode::One => "1.0_f64".to_string(),
        EmlNode::Const(c) => format!("{c}_f64"),
        EmlNode::Var(i) => format!("x{i}"),
        EmlNode::Eml { left, right } => format!(
            "(({}).exp() - ({}).ln())",
            rust_expr(left),
            rust_expr(right)
        ),
    }
}

fn numpy_expr(node: &EmlNode) -> String {
    match node {
        EmlNode::One => "1.0".to_string(),
        EmlNode::Const(c) => format!("{c}"),
        EmlNode::Var(i) => format!("x{i}"),
        EmlNode::Eml { left, right } => format!(
            "(np.exp({}) - np.log({}))",
            numpy_expr(left),
            numpy_expr(right)
        ),
    }
}

fn sympy_expr(node: &EmlNode) -> String {
    match node {
        EmlNode::One => "1".to_string(),
        EmlNode::Const(c) => format!("{c}"),
        EmlNode::Var(i) => format!("x{i}"),
        EmlNode::Eml { left, right } => {
            format!("(exp({}) - log({}))", sympy_expr(left), sympy_expr(right))
        }
    }
}

/// Generate a standalone Rust function `f(x0, x1, ...) -> f64` for the tree.
#[must_use]
pub fn rust_code(tree: &EmlTree) -> String {
    let n = tree.num_vars().max(1);
    let args = (0..n)
        .map(|i| format!("x{i}: f64"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("fn f({args}) -> f64 {{ {} }}", rust_expr(&tree.root))
}

/// Generate a NumPy-compatible Python lambda body for the tree.
#[must_use]
pub fn numpy_code(tree: &EmlTree) -> String {
    let n = tree.num_vars().max(1);
    let args = (0..n)
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("lambda {args}: {}", numpy_expr(&tree.root))
}

/// Generate a SymPy expression string for the tree.
///
/// The result uses SymPy's `exp` / `log` and bare symbols `x0, x1, …`; it parses under
/// `from sympy import exp, log, symbols` (with the `xi` declared as symbols) and can then be
/// `simplify`-ed Python-side. The variables are listed in the doc comment of the emitted code.
#[must_use]
pub fn sympy_code(tree: &EmlTree) -> String {
    let n = tree.num_vars().max(1);
    let syms = (0..n)
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("# symbols('{syms}')\n{}", sympy_expr(&tree.root))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_codegen_for_exp() {
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let code = rust_code(&tree);
        assert!(code.contains("fn f("));
        assert!(code.contains(".exp()"));
    }

    #[test]
    fn numpy_codegen_for_exp() {
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let code = numpy_code(&tree);
        assert!(code.contains("np.exp"));
    }

    #[test]
    fn sympy_codegen_for_exp() {
        let tree = oxieml::Canonical::exp(&EmlTree::var(0));
        let code = sympy_code(&tree);
        assert!(code.contains("exp("));
        assert!(code.contains("log("));
        assert!(code.contains("x0"));
    }
}
