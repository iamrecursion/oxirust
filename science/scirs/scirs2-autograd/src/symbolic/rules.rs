//! Differentiation rules and pattern matching

use super::SymExpr;
use crate::Float;

/// Apply chain rule
pub fn chain_rule<T: Float>(
    outer: &SymExpr<T>,
    inner: &SymExpr<T>,
    inner_deriv: &SymExpr<T>,
) -> SymExpr<T> {
    // d/dx f(g(x)) = f'(g(x)) * g'(x)
    SymExpr::Mul(Box::new(outer.clone()), Box::new(inner_deriv.clone()))
}

/// Apply product rule
pub fn product_rule<T: Float>(
    a: &SymExpr<T>,
    b: &SymExpr<T>,
    a_deriv: &SymExpr<T>,
    b_deriv: &SymExpr<T>,
) -> SymExpr<T> {
    // (a*b)' = a'*b + a*b'
    SymExpr::Add(
        Box::new(SymExpr::Mul(Box::new(a_deriv.clone()), Box::new(b.clone()))),
        Box::new(SymExpr::Mul(Box::new(a.clone()), Box::new(b_deriv.clone()))),
    )
}

/// Apply quotient rule
pub fn quotient_rule<T: Float>(
    a: &SymExpr<T>,
    b: &SymExpr<T>,
    a_deriv: &SymExpr<T>,
    b_deriv: &SymExpr<T>,
) -> SymExpr<T> {
    // (a/b)' = (a'*b - a*b') / b²
    SymExpr::Div(
        Box::new(SymExpr::Sub(
            Box::new(SymExpr::Mul(Box::new(a_deriv.clone()), Box::new(b.clone()))),
            Box::new(SymExpr::Mul(Box::new(a.clone()), Box::new(b_deriv.clone()))),
        )),
        Box::new(SymExpr::Pow(
            Box::new(b.clone()),
            Box::new(SymExpr::Const(T::from(2).expect("Convert 2"))),
        )),
    )
}
