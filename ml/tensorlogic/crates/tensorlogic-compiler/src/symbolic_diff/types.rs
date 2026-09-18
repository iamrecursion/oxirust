//! Public types and internal context for symbolic differentiation.

use tensorlogic_ir::TLExpr;

/// Configuration for symbolic differentiation.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Automatically simplify algebraic identities after differentiation.
    pub simplify_result: bool,
    /// Return an error instead of `Zero` for unsupported expression nodes.
    pub error_on_unsupported: bool,
    /// Maximum recursion depth; guards against exponential expression blowup.
    pub max_expr_depth: usize,
}

impl Default for DiffConfig {
    fn default() -> Self {
        DiffConfig {
            simplify_result: true,
            error_on_unsupported: false,
            max_expr_depth: 50,
        }
    }
}

/// The result of differentiating a single expression.
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// The computed derivative d(expr)/d(var).
    pub derivative: TLExpr,
    /// Whether post-differentiation simplification was applied.
    pub simplified: bool,
    /// Names of expression nodes that were unsupported and fell through to `Zero`.
    pub unsupported_nodes: Vec<String>,
}

/// Error type for symbolic differentiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffError {
    /// Recursion depth exceeded `DiffConfig::max_expr_depth`.
    MaxDepthExceeded,
    /// An unsupported expression was encountered and `error_on_unsupported` was set.
    ExprTooComplex(String),
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffError::MaxDepthExceeded => write!(f, "maximum differentiation depth exceeded"),
            DiffError::ExprTooComplex(msg) => {
                write!(f, "expression too complex or unsupported: {}", msg)
            }
        }
    }
}

impl std::error::Error for DiffError {}

/// Mutable state threaded through the recursive differentiator.
pub(super) struct DiffContext<'a> {
    pub(super) var: String,
    pub(super) config: &'a DiffConfig,
    pub(super) depth: usize,
    pub(super) unsupported_nodes: Vec<String>,
}
