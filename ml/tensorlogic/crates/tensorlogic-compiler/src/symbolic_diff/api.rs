//! Top-level public entry points (differentiate / jacobian).

use tensorlogic_ir::TLExpr;

use super::diff_core::diff_expr;
use super::helpers::simplify_derivative;
use super::types::{DiffConfig, DiffContext, DiffError, DiffResult};

/// Symbolically differentiate `expr` with respect to the variable named `var`.
///
/// # Differentiation rules
///
/// - `d(c)/dx = 0` for any constant `c`
/// - `d(x)/dx = 1`
/// - `d(y)/dx = 0` for `y ≠ x`
/// - Sum rule: `d(a + b)/dx = d(a)/dx + d(b)/dx`
/// - Product rule: `d(a * b)/dx = a * d(b)/dx + b * d(a)/dx`
/// - Quotient rule: `d(a / b)/dx = (d(a)/dx * b − a * d(b)/dx) / b²`
/// - Power rule: `d(a^n)/dx = n * a^(n−1) * d(a)/dx` (when exponent is a constant)
/// - Chain rule applies to transcendental unary functions
/// - Logical AND: `d(AND(a,b))/dx = AND(d(a)/dx, b) OR AND(a, d(b)/dx)`
/// - Logical OR: `d(OR(a,b))/dx = OR(d(a)/dx, d(b)/dx)`
/// - Logical NOT: `d(NOT(a))/dx = NOT(d(a)/dx)`
/// - Implication: expanded as `NOT(a) OR b` before differentiating
/// - Quantifiers: bound variable shadowed; derivative of body is returned
/// - Let-binding: full chain-rule expansion via d(body)/d(bound) * d(value)/dx
///
/// # Errors
///
/// Returns `DiffError::MaxDepthExceeded` if the expression tree exceeds
/// `config.max_expr_depth`. Returns `DiffError::ExprTooComplex` if an
/// unsupported node is encountered and `config.error_on_unsupported` is true.
pub fn differentiate(
    expr: &TLExpr,
    var: &str,
    config: &DiffConfig,
) -> Result<DiffResult, DiffError> {
    let mut ctx = DiffContext {
        var: var.to_string(),
        config,
        depth: 0,
        unsupported_nodes: Vec::new(),
    };
    let derivative = diff_expr(expr, &mut ctx)?;
    let (final_expr, simplified) = if config.simplify_result {
        (simplify_derivative(derivative), true)
    } else {
        (derivative, false)
    };
    Ok(DiffResult {
        derivative: final_expr,
        simplified,
        unsupported_nodes: ctx.unsupported_nodes,
    })
}

/// Compute the Jacobian: differentiate `expr` with respect to each variable in `vars`.
///
/// Returns a vector of `(variable_name, DiffResult)` pairs in the same order as `vars`.
pub fn jacobian(
    expr: &TLExpr,
    vars: &[&str],
    config: &DiffConfig,
) -> Result<Vec<(String, DiffResult)>, DiffError> {
    let mut results = Vec::with_capacity(vars.len());
    for &v in vars {
        let result = differentiate(expr, v, config)?;
        results.push((v.to_string(), result));
    }
    Ok(results)
}
