//! Expression utilities and builders

use super::SymExpr;
use crate::Float;

/// Builder for symbolic expressions
pub struct ExprBuilder<T: Float> {
    expr: SymExpr<T>,
}

impl<T: Float> ExprBuilder<T> {
    /// Create a new expression builder from a variable
    pub fn var(name: impl Into<String>) -> Self {
        Self {
            expr: SymExpr::variable(name),
        }
    }

    /// Create a new expression builder from a constant
    pub fn constant(value: T) -> Self {
        Self {
            expr: SymExpr::constant(value),
        }
    }

    /// Add another expression
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: SymExpr<T>) -> Self {
        Self {
            expr: SymExpr::Add(Box::new(self.expr), Box::new(other)),
        }
    }

    /// Multiply by another expression
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: SymExpr<T>) -> Self {
        Self {
            expr: SymExpr::Mul(Box::new(self.expr), Box::new(other)),
        }
    }

    /// Raise to a power
    pub fn pow(self, exponent: T) -> Self {
        Self {
            expr: SymExpr::Pow(Box::new(self.expr), Box::new(SymExpr::constant(exponent))),
        }
    }

    /// Build the expression
    pub fn build(self) -> SymExpr<T> {
        self.expr
    }
}

/// Helper macro for building symbolic expressions
#[macro_export]
macro_rules! symexpr {
    ($var:ident) => {
        $crate::symbolic::SymExpr::variable(stringify!($var))
    };
    ($val:literal) => {
        $crate::symbolic::SymExpr::constant($val)
    };
}
