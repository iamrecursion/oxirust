//! Helper constructors and the algebraic simplification pass.

use tensorlogic_ir::TLExpr;

/// Build the canonical zero expression.
#[inline]
pub(super) fn zero() -> TLExpr {
    TLExpr::Constant(0.0)
}

/// Build the canonical one expression.
#[inline]
pub(super) fn one() -> TLExpr {
    TLExpr::Constant(1.0)
}

/// Check whether `expr` is a constant equal to `value` (within f64 epsilon).
#[inline]
pub(super) fn is_constant_value(expr: &TLExpr, value: f64) -> bool {
    match expr {
        TLExpr::Constant(v) => (v - value).abs() < f64::EPSILON,
        _ => false,
    }
}

/// Return a symbolic representation of the derivative of a function node.
///
/// For a known named function `f` (zero-arity predicate), the marker is `Pred("d_f", [])`.
/// For a complex/anonymous function, `Pred("d_f", [])` is used as a generic marker.
pub(super) fn derivative_of_function(function: &TLExpr) -> TLExpr {
    match function {
        TLExpr::Pred { name, args } if args.is_empty() => {
            TLExpr::pred(format!("d_{}", name), vec![])
        }
        _ => TLExpr::pred("d_f".to_string(), vec![]),
    }
}

/// Basic algebraic simplification applied to derivative expressions.
///
/// Simplification rules:
/// - `0 + x → x`,  `x + 0 → x`
/// - `0 * x → 0`,  `x * 0 → 0`
/// - `1 * x → x`,  `x * 1 → x`
/// - `x - 0 → x`
/// - `0 - c → -c` (constant folding for arithmetic negation form)
/// - `0 / x → 0`
/// - `x ^ 0 → 1`,  `x ^ 1 → x`
/// - constant folding for pure-constant arithmetic nodes
pub fn simplify_derivative(expr: TLExpr) -> TLExpr {
    match expr {
        TLExpr::Add(l, r) => {
            let l = simplify_derivative(*l);
            let r = simplify_derivative(*r);
            if is_constant_value(&l, 0.0) {
                return r;
            }
            if is_constant_value(&r, 0.0) {
                return l;
            }
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&l, &r) {
                return TLExpr::Constant(a + b);
            }
            TLExpr::Add(Box::new(l), Box::new(r))
        }

        TLExpr::Sub(l, r) => {
            let l = simplify_derivative(*l);
            let r = simplify_derivative(*r);
            if is_constant_value(&r, 0.0) {
                return l;
            }
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&l, &r) {
                return TLExpr::Constant(a - b);
            }
            TLExpr::Sub(Box::new(l), Box::new(r))
        }

        TLExpr::Mul(l, r) => {
            let l = simplify_derivative(*l);
            let r = simplify_derivative(*r);
            if is_constant_value(&l, 0.0) || is_constant_value(&r, 0.0) {
                return TLExpr::Constant(0.0);
            }
            if is_constant_value(&l, 1.0) {
                return r;
            }
            if is_constant_value(&r, 1.0) {
                return l;
            }
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&l, &r) {
                return TLExpr::Constant(a * b);
            }
            TLExpr::Mul(Box::new(l), Box::new(r))
        }

        TLExpr::Div(l, r) => {
            let l = simplify_derivative(*l);
            let r = simplify_derivative(*r);
            if is_constant_value(&l, 0.0) {
                return TLExpr::Constant(0.0);
            }
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&l, &r) {
                if b.abs() > f64::EPSILON {
                    return TLExpr::Constant(a / b);
                }
            }
            TLExpr::Div(Box::new(l), Box::new(r))
        }

        TLExpr::Pow(base, exp) => {
            let base = simplify_derivative(*base);
            let exp = simplify_derivative(*exp);
            if is_constant_value(&exp, 0.0) {
                return TLExpr::Constant(1.0);
            }
            if is_constant_value(&exp, 1.0) {
                return base;
            }
            if let (TLExpr::Constant(b), TLExpr::Constant(e)) = (&base, &exp) {
                return TLExpr::Constant(b.powf(*e));
            }
            TLExpr::Pow(Box::new(base), Box::new(exp))
        }

        TLExpr::And(l, r) => TLExpr::And(
            Box::new(simplify_derivative(*l)),
            Box::new(simplify_derivative(*r)),
        ),
        TLExpr::Or(l, r) => TLExpr::Or(
            Box::new(simplify_derivative(*l)),
            Box::new(simplify_derivative(*r)),
        ),
        TLExpr::Not(inner) => TLExpr::Not(Box::new(simplify_derivative(*inner))),

        other => other,
    }
}
