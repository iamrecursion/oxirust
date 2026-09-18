//! Pattern type for TLExpr pattern matching.
//!
//! Provides the `MatchPattern` enum used in `TLExpr::Match` arms. Only concrete
//! constant patterns and a wildcard are supported (Design A — no variable binding).

use std::fmt;

use serde::{Deserialize, Serialize};

/// A pattern used in a `TLExpr::Match` arm.
///
/// Patterns are intentionally minimal (Design A):
/// - `ConstSymbol(String)` — match a symbol literal by name.
/// - `ConstNumber(f64)` — match a numeric literal by value.
/// - `Wildcard` — match anything; must be the last arm in a `Match`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MatchPattern {
    /// Match a symbol literal (lowered to `Eq(scrutinee, SymbolLiteral(s))`).
    ConstSymbol(String),
    /// Match a numeric literal (lowered to `Eq(scrutinee, Constant(n))`).
    ConstNumber(f64),
    /// Wildcard — matches anything. Must be the last arm.
    Wildcard,
}

impl fmt::Display for MatchPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConstSymbol(s) => write!(f, ":{s}"),
            Self::ConstNumber(n) => write!(f, "{n}"),
            Self::Wildcard => write!(f, "_"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_patterns() {
        assert_eq!(MatchPattern::ConstSymbol("ok".into()).to_string(), ":ok");
        assert_eq!(MatchPattern::ConstNumber(2.71).to_string(), "2.71");
        assert_eq!(MatchPattern::Wildcard.to_string(), "_");
    }
}
