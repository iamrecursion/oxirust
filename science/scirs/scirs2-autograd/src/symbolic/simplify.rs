//! Expression simplification rules

use super::SymExpr;
use crate::Float;

/// Simplify a symbolic expression
pub fn simplify_expr<T: Float>(expr: &SymExpr<T>) -> SymExpr<T> {
    match expr {
        // Constant stays constant
        SymExpr::Const(c) => SymExpr::Const(*c),

        // Variable stays variable
        SymExpr::Var(name) => SymExpr::Var(name.clone()),

        // Addition simplification
        SymExpr::Add(a, b) => {
            let a_simp = simplify_expr(a);
            let b_simp = simplify_expr(b);

            match (&a_simp, &b_simp) {
                // 0 + x = x
                (SymExpr::Const(c), _) if c.abs() < T::epsilon() => b_simp,
                // x + 0 = x
                (_, SymExpr::Const(c)) if c.abs() < T::epsilon() => a_simp,
                // c1 + c2 = c1+c2
                (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(*c1 + *c2),
                // Default: keep as is
                _ => SymExpr::Add(Box::new(a_simp), Box::new(b_simp)),
            }
        }

        // Multiplication simplification
        SymExpr::Mul(a, b) => {
            let a_simp = simplify_expr(a);
            let b_simp = simplify_expr(b);

            match (&a_simp, &b_simp) {
                // 0 * x = 0
                (SymExpr::Const(c), _) if c.abs() < T::epsilon() => SymExpr::Const(T::zero()),
                // x * 0 = 0
                (_, SymExpr::Const(c)) if c.abs() < T::epsilon() => SymExpr::Const(T::zero()),
                // 1 * x = x
                (SymExpr::Const(c), _) if (*c - T::one()).abs() < T::epsilon() => b_simp,
                // x * 1 = x
                (_, SymExpr::Const(c)) if (*c - T::one()).abs() < T::epsilon() => a_simp,
                // c1 * c2 = c1*c2
                (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(*c1 * *c2),
                // Default: keep as is
                _ => SymExpr::Mul(Box::new(a_simp), Box::new(b_simp)),
            }
        }

        // Subtraction simplification
        SymExpr::Sub(a, b) => {
            let a_simp = simplify_expr(a);
            let b_simp = simplify_expr(b);

            match (&a_simp, &b_simp) {
                // x - 0 = x
                (_, SymExpr::Const(c)) if c.abs() < T::epsilon() => a_simp,
                // x - x = 0
                _ if a_simp == b_simp => SymExpr::Const(T::zero()),
                // c1 - c2 = c1-c2
                (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(*c1 - *c2),
                // Default: keep as is
                _ => SymExpr::Sub(Box::new(a_simp), Box::new(b_simp)),
            }
        }

        // Division simplification
        SymExpr::Div(a, b) => {
            let a_simp = simplify_expr(a);
            let b_simp = simplify_expr(b);

            match (&a_simp, &b_simp) {
                // 0 / x = 0
                (SymExpr::Const(c), _) if c.abs() < T::epsilon() => SymExpr::Const(T::zero()),
                // x / 1 = x
                (_, SymExpr::Const(c)) if (*c - T::one()).abs() < T::epsilon() => a_simp,
                // c1 / c2 = c1/c2
                (SymExpr::Const(c1), SymExpr::Const(c2)) if c2.abs() > T::epsilon() => {
                    SymExpr::Const(*c1 / *c2)
                }
                // Default: keep as is
                _ => SymExpr::Div(Box::new(a_simp), Box::new(b_simp)),
            }
        }

        // Power simplification
        SymExpr::Pow(a, b) => {
            let a_simp = simplify_expr(a);
            let b_simp = simplify_expr(b);

            match (&a_simp, &b_simp) {
                // x ^ 0 = 1
                (_, SymExpr::Const(c)) if c.abs() < T::epsilon() => SymExpr::Const(T::one()),
                // x ^ 1 = x
                (_, SymExpr::Const(c)) if (*c - T::one()).abs() < T::epsilon() => a_simp,
                // 0 ^ x = 0 (for x > 0)
                (SymExpr::Const(c1), SymExpr::Const(c2))
                    if c1.abs() < T::epsilon() && *c2 > T::zero() =>
                {
                    SymExpr::Const(T::zero())
                }
                // c1 ^ c2 = c1^c2
                (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(c1.powf(*c2)),
                // Default: keep as is
                _ => SymExpr::Pow(Box::new(a_simp), Box::new(b_simp)),
            }
        }

        // Simplify argument and rebuild
        SymExpr::Exp(a) => SymExpr::Exp(Box::new(simplify_expr(a))),
        SymExpr::Log(a) => SymExpr::Log(Box::new(simplify_expr(a))),
        SymExpr::Sin(a) => SymExpr::Sin(Box::new(simplify_expr(a))),
        SymExpr::Cos(a) => SymExpr::Cos(Box::new(simplify_expr(a))),
        SymExpr::Tanh(a) => SymExpr::Tanh(Box::new(simplify_expr(a))),
    }
}
