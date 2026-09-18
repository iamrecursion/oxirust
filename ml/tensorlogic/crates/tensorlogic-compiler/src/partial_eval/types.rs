//! Public types for the partial evaluator: [`PEValue`], [`PEEnv`], [`PEStats`],
//! [`PEConfig`], and [`PEResult`].

use std::collections::HashMap;

use tensorlogic_ir::TLExpr;

// ── PEValue ───────────────────────────────────────────────────────────────────

/// A value in partial evaluation — either a concrete numeric scalar, a concrete
/// boolean, or still symbolic (cannot be reduced further with current environment).
#[derive(Debug, Clone)]
pub enum PEValue {
    /// A concrete floating-point scalar.
    Concrete(f64),
    /// A concrete boolean truth value.
    Boolean(bool),
    /// Still symbolic — cannot be further reduced.
    Symbolic(TLExpr),
}

impl PEValue {
    /// Returns `true` when the value is fully concrete (numeric or boolean).
    pub fn is_concrete(&self) -> bool {
        matches!(self, PEValue::Concrete(_) | PEValue::Boolean(_))
    }

    /// Extracts the `f64` if this is a [`PEValue::Concrete`], otherwise `None`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PEValue::Concrete(v) => Some(*v),
            PEValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            PEValue::Symbolic(_) => None,
        }
    }

    /// Extracts the `bool` if this is a [`PEValue::Boolean`], otherwise `None`.
    /// A `Concrete(1.0)` is treated as `true`; `Concrete(0.0)` as `false`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PEValue::Boolean(b) => Some(*b),
            PEValue::Concrete(v) => Some(*v != 0.0),
            PEValue::Symbolic(_) => None,
        }
    }

    /// Convert this value back into a [`TLExpr`].
    pub fn to_expr(&self) -> TLExpr {
        match self {
            PEValue::Concrete(v) => TLExpr::Constant(*v),
            PEValue::Boolean(true) => TLExpr::Constant(1.0),
            PEValue::Boolean(false) => TLExpr::Constant(0.0),
            PEValue::Symbolic(e) => e.clone(),
        }
    }

    /// Build a `PEValue` from a `TLExpr` node: if it is `Constant` it becomes
    /// `Concrete`, otherwise `Symbolic`.
    pub(super) fn from_expr(expr: TLExpr) -> Self {
        match expr {
            TLExpr::Constant(v) => PEValue::Concrete(v),
            other => PEValue::Symbolic(other),
        }
    }
}

// ── PEEnv ────────────────────────────────────────────────────────────────────

/// Partial-evaluation environment: a map from variable name to its known [`PEValue`].
///
/// Variables are represented as zero-arity predicates in TLExpr (e.g. `Pred { name: "x", args: [] }`).
/// This environment records which of those logical variables have been concretised.
#[derive(Debug, Clone, Default)]
pub struct PEEnv {
    bindings: HashMap<String, PEValue>,
}

impl PEEnv {
    /// Create an empty environment.
    pub fn new() -> Self {
        PEEnv {
            bindings: HashMap::new(),
        }
    }

    /// Builder-style helper: bind a floating-point variable.
    pub fn with_f64(mut self, var: impl Into<String>, val: f64) -> Self {
        self.bindings.insert(var.into(), PEValue::Concrete(val));
        self
    }

    /// Builder-style helper: bind a boolean variable.
    pub fn with_bool(mut self, var: impl Into<String>, val: bool) -> Self {
        self.bindings.insert(var.into(), PEValue::Boolean(val));
        self
    }

    /// Mutably bind a floating-point variable.
    pub fn bind_f64(&mut self, var: impl Into<String>, val: f64) {
        self.bindings.insert(var.into(), PEValue::Concrete(val));
    }

    /// Mutably bind a boolean variable.
    pub fn bind_bool(&mut self, var: impl Into<String>, val: bool) {
        self.bindings.insert(var.into(), PEValue::Boolean(val));
    }

    /// Look up a variable in the environment.
    pub fn lookup(&self, var: &str) -> Option<&PEValue> {
        self.bindings.get(var)
    }

    /// Return a new environment extended with one additional binding.
    /// The original environment is not mutated.
    pub fn extend(&self, var: impl Into<String>, val: PEValue) -> PEEnv {
        let mut new_env = self.clone();
        new_env.bindings.insert(var.into(), val);
        new_env
    }

    /// Number of bindings in this environment.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns `true` when the environment has no bindings.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

// ── PEStats ──────────────────────────────────────────────────────────────────

/// Statistics accumulated during a partial evaluation run.
#[derive(Debug, Default, Clone)]
pub struct PEStats {
    /// Total expression nodes visited.
    pub nodes_visited: usize,
    /// Nodes that were replaced by a concrete value.
    pub nodes_reduced: usize,
    /// `Let` bindings that were completely eliminated (body inlined).
    pub lets_inlined: usize,
    /// Dead branches pruned due to a known boolean condition.
    pub branches_pruned: usize,
}

impl PEStats {
    /// Fraction of visited nodes that were reduced to concrete values.
    /// Returns `0.0` when `nodes_visited == 0`.
    pub fn reduction_rate(&self) -> f64 {
        if self.nodes_visited == 0 {
            return 0.0;
        }
        self.nodes_reduced as f64 / self.nodes_visited as f64
    }
}

// ── PEConfig ─────────────────────────────────────────────────────────────────

/// Configuration governing which reductions are attempted during partial evaluation.
#[derive(Debug, Clone)]
pub struct PEConfig {
    /// Maximum recursion depth before the evaluator stops descending.
    pub max_depth: usize,
    /// If `true`, inline `Let` bindings whose bound expression reduces to a concrete value.
    pub inline_lets: bool,
    /// If `true`, prune dead branches in logical operators when one operand is
    /// a known boolean constant.
    pub prune_branches: bool,
    /// If `true`, fold arithmetic operations when both operands are concrete.
    pub fold_arithmetic: bool,
    /// If `true`, fold logical operations on known boolean operands.
    pub fold_logic: bool,
}

impl Default for PEConfig {
    fn default() -> Self {
        PEConfig {
            max_depth: 200,
            inline_lets: true,
            prune_branches: true,
            fold_arithmetic: true,
            fold_logic: true,
        }
    }
}

// ── PEResult ─────────────────────────────────────────────────────────────────

/// The result of partially evaluating an expression.
pub struct PEResult {
    /// The residual (simplified) expression.
    pub expr: TLExpr,
    /// Statistics accumulated during evaluation.
    pub stats: PEStats,
    /// Names of variables still free (unbound) in the output expression.
    pub residual_vars: Vec<String>,
}
