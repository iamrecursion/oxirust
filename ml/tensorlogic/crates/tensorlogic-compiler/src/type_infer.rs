//! Unification-based type inference for TLExpr subexpressions.
//!
//! This module assigns semantic types to every subexpression in a [`TLExpr`]
//! tree using a Hindley–Milner–style constraint solver (Robinson unification).
//! The inferred types drive early type-mismatch detection and annotate nodes
//! for downstream optimisation passes.
//!
//! # Type Lattice
//!
//! ```text
//!   Bool        – logical / Boolean expression
//!   Numeric     – real-valued / f64 arithmetic
//!   Relation(n) – n-ary predicate / relation
//!   Set         – set of values
//!   Fuzzy       – fuzzy truth value ∈ [0, 1]
//!   Probabilistic – probability value ∈ [0, 1]
//!   Var(id)     – unification placeholder
//!   Unknown     – could not be determined
//! ```
//!
//! # Quick Start
//!
//! ```rust
//! use tensorlogic_compiler::type_infer::{infer_type, TypeEnv, TLType};
//! use tensorlogic_ir::TLExpr;
//!
//! let expr = TLExpr::and(
//!     TLExpr::pred("p", vec![]),
//!     TLExpr::pred("q", vec![]),
//! );
//! let result = infer_type(&expr, &TypeEnv::new());
//! assert_eq!(result.typed_expr.ty, TLType::Bool);
//! ```

use std::collections::HashMap;
use std::fmt;

use tensorlogic_ir::TLExpr;

// ─────────────────────────────────────────────────────────────────────────────
// TLType
// ─────────────────────────────────────────────────────────────────────────────

/// The semantic type of a TLExpr subexpression.
#[derive(Debug, Clone, PartialEq)]
pub enum TLType {
    /// Logical / Boolean expression (true / false).
    Bool,
    /// Real-valued / f64 arithmetic expression.
    Numeric,
    /// n-ary relation (predicate with n arguments).
    Relation(usize),
    /// A set of values.
    Set,
    /// Fuzzy truth value ∈ \[0, 1\].
    Fuzzy,
    /// Probability value ∈ \[0, 1\].
    Probabilistic,
    /// Unification placeholder (type variable with unique id).
    Var(usize),
    /// Type could not be determined (non-fatal).
    Unknown,
}

impl TLType {
    /// Returns `true` when the type contains no [`TLType::Var`] placeholders.
    pub fn is_ground(&self) -> bool {
        !matches!(self, TLType::Var(_))
    }

    /// Human-readable name for the type (no allocation).
    pub fn display_name(&self) -> &'static str {
        match self {
            TLType::Bool => "Bool",
            TLType::Numeric => "Numeric",
            TLType::Relation(_) => "Relation",
            TLType::Set => "Set",
            TLType::Fuzzy => "Fuzzy",
            TLType::Probabilistic => "Probabilistic",
            TLType::Var(_) => "Var",
            TLType::Unknown => "Unknown",
        }
    }
}

impl fmt::Display for TLType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TLType::Relation(n) => write!(f, "Relation({})", n),
            TLType::Var(id) => write!(f, "Var({})", id),
            other => write!(f, "{}", other.display_name()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TyVarCounter – fresh type-variable generator
// ─────────────────────────────────────────────────────────────────────────────

/// Monotone counter that issues fresh type-variable identifiers.
#[derive(Debug, Default)]
pub struct TyVarCounter {
    next: usize,
}

impl TyVarCounter {
    /// Create a new counter starting at 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a fresh [`TLType::Var`] and advance the counter.
    pub fn fresh(&mut self) -> TLType {
        let id = self.next;
        self.next += 1;
        TLType::Var(id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Substitution – unification map (type-var id → resolved type)
// ─────────────────────────────────────────────────────────────────────────────

/// Maps type-variable identifiers to their resolved types.
///
/// Supports path compression via [`Substitution::apply`].
#[derive(Debug, Default, Clone)]
pub struct Substitution {
    map: HashMap<usize, TLType>,
}

impl Substitution {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind type variable `var` to `ty`.
    ///
    /// Overwrites any existing binding.
    pub fn bind(&mut self, var: usize, ty: TLType) {
        self.map.insert(var, ty);
    }

    /// Look up the immediate binding of type variable `var`.
    pub fn lookup(&self, var: usize) -> Option<&TLType> {
        self.map.get(&var)
    }

    /// Chase a type through all variable bindings and return the fully applied
    /// (shallowest ground) type.  Does not recurse into `Relation(n)` contents
    /// since `TLType` has no nested type parameters there.
    pub fn apply(&self, ty: &TLType) -> TLType {
        match ty {
            TLType::Var(id) => {
                // Chase the chain to its end.
                let mut current_id = *id;
                let mut visited = Vec::new(); // cycle guard
                loop {
                    if visited.contains(&current_id) {
                        // Cycle detected; return the variable as-is.
                        return TLType::Var(current_id);
                    }
                    visited.push(current_id);
                    match self.map.get(&current_id) {
                        None => return TLType::Var(current_id),
                        Some(TLType::Var(next_id)) => {
                            current_id = *next_id;
                        }
                        Some(other) => return other.clone(),
                    }
                }
            }
            other => other.clone(),
        }
    }

    /// Number of bindings in this substitution.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` when the substitution has no bindings.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TypeInferError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that arise during type inference.
#[derive(Debug)]
pub enum TypeInferError {
    /// Two types could not be unified.
    UnificationFailed { expected: String, got: String },
    /// A variable name was used but has no binding in the environment.
    UnboundVariable(String),
    /// Occurs check: a type variable appears inside the type it would be bound to.
    OccursCheck(usize, String),
    /// A predicate was applied to the wrong number of arguments.
    ArityMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for TypeInferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeInferError::UnificationFailed { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            TypeInferError::UnboundVariable(name) => {
                write!(f, "unbound variable: {}", name)
            }
            TypeInferError::OccursCheck(id, ty) => {
                write!(
                    f,
                    "occurs check failed: type variable Var({}) occurs in {}",
                    id, ty
                )
            }
            TypeInferError::ArityMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "arity mismatch for '{}': expected {} args, got {}",
                    name, expected, got
                )
            }
        }
    }
}

impl std::error::Error for TypeInferError {}

// ─────────────────────────────────────────────────────────────────────────────
// TypeEnv – variable-name → type bindings
// ─────────────────────────────────────────────────────────────────────────────

/// Maps logic variable names (strings) to their inferred [`TLType`].
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    bindings: HashMap<String, TLType>,
}

impl TypeEnv {
    /// Create an empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder helper: add a binding and return `self`.
    pub fn with(mut self, var: impl Into<String>, ty: TLType) -> Self {
        self.bindings.insert(var.into(), ty);
        self
    }

    /// Insert or update a binding in place.
    pub fn bind(&mut self, var: impl Into<String>, ty: TLType) {
        self.bindings.insert(var.into(), ty);
    }

    /// Look up the type of a variable.
    pub fn lookup(&self, var: &str) -> Option<&TLType> {
        self.bindings.get(var)
    }

    /// Non-destructively extend this environment with one new binding.
    ///
    /// The original environment is unchanged; a new [`TypeEnv`] is returned.
    pub fn extend(&self, var: impl Into<String>, ty: TLType) -> TypeEnv {
        let mut new_env = self.clone();
        new_env.bind(var, ty);
        new_env
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TypedExpr – annotated expression tree
// ─────────────────────────────────────────────────────────────────────────────

/// A [`TLExpr`] node annotated with its inferred type and typed children.
#[derive(Debug, Clone)]
pub struct TypedExpr {
    /// The original expression node (shallow clone – no recursive boxing).
    pub expr: TLExpr,
    /// The inferred type of this node.
    pub ty: TLType,
    /// Type-annotated direct children of this node.
    pub children: Vec<TypedExpr>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TypeInferResult – top-level output
// ─────────────────────────────────────────────────────────────────────────────

/// The result produced by [`infer_type`].
pub struct TypeInferResult {
    /// The root annotated expression.
    pub typed_expr: TypedExpr,
    /// The final unification substitution.
    pub subst: Substitution,
    /// Number of type variables that were resolved to ground types.
    pub inferred_vars: usize,
    /// Non-fatal type errors (e.g., `Unknown` nodes).
    pub errors: Vec<TypeInferError>,
}

// ─────────────────────────────────────────────────────────────────────────────
// occurs_check
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` when type variable `id` appears (directly) in `ty`
/// after chasing the substitution.  Used to prevent infinite types.
fn occurs_in(id: usize, ty: &TLType, subst: &Substitution) -> bool {
    let resolved = subst.apply(ty);
    match resolved {
        TLType::Var(other_id) => other_id == id,
        // TLType variants are all flat (no nested TLType args), so no recursion needed.
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// unify – Robinson unification
// ─────────────────────────────────────────────────────────────────────────────

/// Unify two types under the current substitution using Robinson's algorithm.
///
/// # Errors
///
/// Returns [`TypeInferError::UnificationFailed`] when the types are
/// incompatible, or [`TypeInferError::OccursCheck`] when a type variable
/// would need to be bound to a type containing itself.
pub fn unify(t1: &TLType, t2: &TLType, subst: &mut Substitution) -> Result<(), TypeInferError> {
    let a = subst.apply(t1);
    let b = subst.apply(t2);

    match (&a, &b) {
        // Identical ground types: always OK.
        (TLType::Bool, TLType::Bool)
        | (TLType::Numeric, TLType::Numeric)
        | (TLType::Set, TLType::Set)
        | (TLType::Fuzzy, TLType::Fuzzy)
        | (TLType::Probabilistic, TLType::Probabilistic)
        | (TLType::Unknown, TLType::Unknown) => Ok(()),

        // Relations unify only when they have the same arity.
        (TLType::Relation(n), TLType::Relation(m)) => {
            if n == m {
                Ok(())
            } else {
                Err(TypeInferError::UnificationFailed {
                    expected: format!("Relation({})", n),
                    got: format!("Relation({})", m),
                })
            }
        }

        // Var(id) = Var(id): trivially OK.
        (TLType::Var(id1), TLType::Var(id2)) if id1 == id2 => Ok(()),

        // Bind a type variable (occurs check first).
        (TLType::Var(id), other) => {
            if occurs_in(*id, other, subst) {
                Err(TypeInferError::OccursCheck(*id, other.to_string()))
            } else {
                subst.bind(*id, other.clone());
                Ok(())
            }
        }
        (other, TLType::Var(id)) => {
            if occurs_in(*id, other, subst) {
                Err(TypeInferError::OccursCheck(*id, other.to_string()))
            } else {
                subst.bind(*id, other.clone());
                Ok(())
            }
        }

        // Anything else is a genuine mismatch.
        (lhs, rhs) => Err(TypeInferError::UnificationFailed {
            expected: lhs.to_string(),
            got: rhs.to_string(),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// infer – recursive type inference
// ─────────────────────────────────────────────────────────────────────────────

/// Infer the type of `expr` in environment `env`, updating `subst` in place.
///
/// Returns the inferred type or a [`TypeInferError`].
pub fn infer(
    expr: &TLExpr,
    env: &TypeEnv,
    subst: &mut Substitution,
    counter: &mut TyVarCounter,
) -> Result<TLType, TypeInferError> {
    match expr {
        // ── Numeric literals ──────────────────────────────────────────────
        TLExpr::Constant(_) => Ok(TLType::Numeric),

        // ── Predicates ───────────────────────────────────────────────────
        TLExpr::Pred { name: _, args } => {
            if args.is_empty() {
                Ok(TLType::Bool)
            } else {
                Ok(TLType::Relation(args.len()))
            }
        }

        // ── Boolean connectives ───────────────────────────────────────────
        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            let tl = infer(l, env, subst, counter)?;
            unify(&tl, &TLType::Bool, subst)?;
            let tr = infer(r, env, subst, counter)?;
            unify(&tr, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        TLExpr::Not(inner) => {
            let ti = infer(inner, env, subst, counter)?;
            unify(&ti, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        // ── Quantifiers ───────────────────────────────────────────────────
        TLExpr::ForAll {
            var: _,
            domain: _,
            body,
        }
        | TLExpr::Exists {
            var: _,
            domain: _,
            body,
        } => {
            let tb = infer(body, env, subst, counter)?;
            unify(&tb, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        // ── Soft quantifiers → Fuzzy ──────────────────────────────────────
        TLExpr::SoftForAll {
            var: _,
            domain: _,
            body,
            temperature: _,
        }
        | TLExpr::SoftExists {
            var: _,
            domain: _,
            body,
            temperature: _,
        } => {
            let _ = infer(body, env, subst, counter)?;
            Ok(TLType::Fuzzy)
        }

        // ── Arithmetic (binary) ────────────────────────────────────────────
        TLExpr::Add(l, r)
        | TLExpr::Sub(l, r)
        | TLExpr::Mul(l, r)
        | TLExpr::Div(l, r)
        | TLExpr::Pow(l, r)
        | TLExpr::Mod(l, r)
        | TLExpr::Min(l, r)
        | TLExpr::Max(l, r) => {
            let tl = infer(l, env, subst, counter)?;
            unify(&tl, &TLType::Numeric, subst)?;
            let tr = infer(r, env, subst, counter)?;
            unify(&tr, &TLType::Numeric, subst)?;
            Ok(TLType::Numeric)
        }

        // ── Arithmetic (unary) ─────────────────────────────────────────────
        TLExpr::Abs(e)
        | TLExpr::Floor(e)
        | TLExpr::Ceil(e)
        | TLExpr::Round(e)
        | TLExpr::Sqrt(e)
        | TLExpr::Exp(e)
        | TLExpr::Log(e)
        | TLExpr::Sin(e)
        | TLExpr::Cos(e)
        | TLExpr::Tan(e) => {
            let te = infer(e, env, subst, counter)?;
            unify(&te, &TLType::Numeric, subst)?;
            Ok(TLType::Numeric)
        }

        // ── Comparison → Bool (operands Numeric or unifiable to fresh var) ──
        TLExpr::Eq(l, r)
        | TLExpr::Lt(l, r)
        | TLExpr::Gt(l, r)
        | TLExpr::Lte(l, r)
        | TLExpr::Gte(l, r) => {
            let tl = infer(l, env, subst, counter)?;
            let tr = infer(r, env, subst, counter)?;
            // Both sides must have the same type; use a fresh var if needed.
            let fresh = counter.fresh();
            unify(&tl, &fresh, subst)?;
            unify(&tr, &fresh, subst)?;
            Ok(TLType::Bool)
        }

        // ── Conditional ────────────────────────────────────────────────────
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let tc = infer(condition, env, subst, counter)?;
            unify(&tc, &TLType::Bool, subst)?;
            let tt = infer(then_branch, env, subst, counter)?;
            let te = infer(else_branch, env, subst, counter)?;
            unify(&tt, &te, subst)?;
            Ok(subst.apply(&tt))
        }

        // ── Let binding ────────────────────────────────────────────────────
        TLExpr::Let { var, value, body } => {
            let tv = infer(value, env, subst, counter)?;
            let extended = env.extend(var.clone(), tv);
            infer(body, &extended, subst, counter)
        }

        // ── Fuzzy logic ────────────────────────────────────────────────────
        TLExpr::TNorm {
            kind: _,
            left,
            right,
        }
        | TLExpr::TCoNorm {
            kind: _,
            left,
            right,
        } => {
            let _ = infer(left, env, subst, counter)?;
            let _ = infer(right, env, subst, counter)?;
            Ok(TLType::Fuzzy)
        }

        TLExpr::FuzzyNot { kind: _, expr } => {
            let _ = infer(expr, env, subst, counter)?;
            Ok(TLType::Fuzzy)
        }

        TLExpr::FuzzyImplication {
            kind: _,
            premise,
            conclusion,
        } => {
            let _ = infer(premise, env, subst, counter)?;
            let _ = infer(conclusion, env, subst, counter)?;
            Ok(TLType::Fuzzy)
        }

        // ── Probabilistic ──────────────────────────────────────────────────
        TLExpr::WeightedRule { weight: _, rule } => {
            let _ = infer(rule, env, subst, counter)?;
            Ok(TLType::Probabilistic)
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            // Each alternative expression is inferred but we don't unify them;
            // the overall type is Probabilistic.
            for (_, alt_expr) in alternatives {
                let _ = infer(alt_expr, env, subst, counter)?;
            }
            Ok(TLType::Probabilistic)
        }

        // ── Temporal logic (LTL) ───────────────────────────────────────────
        TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner)
        | TLExpr::Box(inner)
        | TLExpr::Diamond(inner) => {
            let ti = infer(inner, env, subst, counter)?;
            unify(&ti, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        TLExpr::Until { before, after }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => {
            let tb = infer(before, env, subst, counter)?;
            unify(&tb, &TLType::Bool, subst)?;
            let ta = infer(after, env, subst, counter)?;
            unify(&ta, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        // ── Score / wrapping ───────────────────────────────────────────────
        TLExpr::Score(inner) => infer(inner, env, subst, counter),

        // ── Aggregation ────────────────────────────────────────────────────
        TLExpr::Aggregate {
            op: _,
            var: _,
            domain: _,
            body,
            group_by: _,
        } => {
            let _ = infer(body, env, subst, counter)?;
            Ok(TLType::Numeric)
        }

        // ── Set operations → Set ───────────────────────────────────────────
        TLExpr::SetUnion { left, right }
        | TLExpr::SetIntersection { left, right }
        | TLExpr::SetDifference { left, right } => {
            let tl = infer(left, env, subst, counter)?;
            unify(&tl, &TLType::Set, subst)?;
            let tr = infer(right, env, subst, counter)?;
            unify(&tr, &TLType::Set, subst)?;
            Ok(TLType::Set)
        }

        TLExpr::SetCardinality { set } => {
            let ts = infer(set, env, subst, counter)?;
            unify(&ts, &TLType::Set, subst)?;
            Ok(TLType::Numeric)
        }

        TLExpr::EmptySet => Ok(TLType::Set),

        TLExpr::SetComprehension {
            var: _,
            domain: _,
            condition,
        } => {
            let tc = infer(condition, env, subst, counter)?;
            unify(&tc, &TLType::Bool, subst)?;
            Ok(TLType::Set)
        }

        TLExpr::SetMembership { element, set } => {
            let _ = infer(element, env, subst, counter)?;
            let ts = infer(set, env, subst, counter)?;
            unify(&ts, &TLType::Set, subst)?;
            Ok(TLType::Bool)
        }

        // ── Counting quantifiers → Bool ────────────────────────────────────
        TLExpr::CountingExists {
            var: _,
            domain: _,
            body,
            min_count: _,
        }
        | TLExpr::CountingForAll {
            var: _,
            domain: _,
            body,
            min_count: _,
        }
        | TLExpr::ExactCount {
            var: _,
            domain: _,
            body,
            count: _,
        }
        | TLExpr::Majority {
            var: _,
            domain: _,
            body,
        } => {
            let tb = infer(body, env, subst, counter)?;
            unify(&tb, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        // ── Fixed-point operators ──────────────────────────────────────────
        TLExpr::LeastFixpoint { var: _, body } | TLExpr::GreatestFixpoint { var: _, body } => {
            let tb = infer(body, env, subst, counter)?;
            // Fixed-point body is typically Bool; return its type.
            Ok(tb)
        }

        // ── Higher-order ───────────────────────────────────────────────────
        TLExpr::Lambda {
            var: _,
            var_type: _,
            body,
        } => {
            // Lambda itself is treated as returning whatever the body returns.
            infer(body, env, subst, counter)
        }

        TLExpr::Apply { function, argument } => {
            // We don't have a full arrow type, so we infer each side and return Unknown.
            let _ = infer(function, env, subst, counter)?;
            let _ = infer(argument, env, subst, counter)?;
            Ok(TLType::Unknown)
        }

        // ── Modal / hybrid logic ───────────────────────────────────────────
        TLExpr::Nominal { name: _ } => Ok(TLType::Bool),

        TLExpr::At {
            nominal: _,
            formula,
        } => {
            let tf = infer(formula, env, subst, counter)?;
            unify(&tf, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
            let tf = infer(formula, env, subst, counter)?;
            unify(&tf, &TLType::Bool, subst)?;
            Ok(TLType::Bool)
        }

        // ── Constraint programming ─────────────────────────────────────────
        TLExpr::AllDifferent { variables: _ } => Ok(TLType::Bool),

        TLExpr::GlobalCardinality {
            variables: _,
            values,
            min_occurrences: _,
            max_occurrences: _,
        } => {
            for v in values {
                let _ = infer(v, env, subst, counter)?;
            }
            Ok(TLType::Bool)
        }

        // ── Abductive reasoning ────────────────────────────────────────────
        TLExpr::Abducible { name: _, cost: _ } => Ok(TLType::Bool),

        TLExpr::Explain { formula } => infer(formula, env, subst, counter),

        // ── Pattern matching ───────────────────────────────────────────────
        TLExpr::SymbolLiteral(_) => Ok(TLType::Unknown),

        TLExpr::Match { scrutinee, arms } => {
            // All arm bodies must unify to the same type
            let _scrutinee_ty = infer(scrutinee, env, subst, counter)?;
            let mut result_ty: Option<TLType> = None;
            for (_, body) in arms {
                let body_ty = infer(body, env, subst, counter)?;
                match &result_ty {
                    None => result_ty = Some(body_ty),
                    Some(rt) => {
                        unify(rt, &body_ty, subst)?;
                    }
                }
            }
            Ok(result_ty.unwrap_or(TLType::Bool))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// infer_type – top-level entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level type inference entry point.
///
/// Runs inference on `expr` in the given environment, collects any non-fatal
/// errors, and returns a [`TypeInferResult`] containing the annotated tree,
/// final substitution, and diagnostics.
pub fn infer_type(expr: &TLExpr, env: &TypeEnv) -> TypeInferResult {
    let mut subst = Substitution::new();
    let mut counter = TyVarCounter::new();
    let mut errors: Vec<TypeInferError> = Vec::new();

    let ty = match infer(expr, env, &mut subst, &mut counter) {
        Ok(t) => t,
        Err(e) => {
            errors.push(e);
            TLType::Unknown
        }
    };

    // Count resolved vars (bindings in the substitution that map to ground types).
    let inferred_vars = subst.map.values().filter(|v| v.is_ground()).count();

    // Build a minimal annotated tree for the root.
    let annotated = annotate_with(expr, env, &subst, &mut TyVarCounter::new(), &mut errors);

    // Override the root type with the directly inferred (applied) type so it is
    // always as resolved as possible.
    let root_ty = subst.apply(&ty);

    let typed_expr = TypedExpr {
        expr: expr.clone(),
        ty: root_ty,
        children: annotated.children,
    };

    TypeInferResult {
        typed_expr,
        subst,
        inferred_vars,
        errors,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// annotate – build a fully annotated TypedExpr tree
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`TypedExpr`] tree by running inference at every node and
/// annotating each with its resolved type.
///
/// # Errors
///
/// Returns the first hard unification error encountered, if any.
pub fn annotate(expr: &TLExpr, env: &TypeEnv) -> Result<TypedExpr, TypeInferError> {
    let mut subst = Substitution::new();
    let mut counter = TyVarCounter::new();
    let mut dummy_errors = Vec::new();
    let typed = annotate_with(expr, env, &subst, &mut counter, &mut dummy_errors);

    // Run a second pass to resolve types that were unified after the first traversal.
    // Re-infer for a definitive substitution.
    let ty = infer(expr, env, &mut subst, &mut counter)?;
    let resolved_ty = subst.apply(&ty);

    Ok(TypedExpr {
        expr: typed.expr,
        ty: resolved_ty,
        children: typed.children,
    })
}

/// Internal recursive helper: annotate `expr` using an already-computed
/// `subst` snapshot and a fresh `counter`.  Non-fatal errors go into `errors`.
fn annotate_with(
    expr: &TLExpr,
    env: &TypeEnv,
    subst: &Substitution,
    counter: &mut TyVarCounter,
    errors: &mut Vec<TypeInferError>,
) -> TypedExpr {
    // We infer with a *local* clone of subst so children can see each other's
    // bindings produced during this annotation pass.
    let mut local_subst = subst.clone();

    let ty = match infer(expr, env, &mut local_subst, counter) {
        Ok(t) => local_subst.apply(&t),
        Err(e) => {
            errors.push(e);
            TLType::Unknown
        }
    };

    // Recursively annotate children.
    let children = collect_children(expr)
        .into_iter()
        .map(|child| annotate_with(child, env, &local_subst, counter, errors))
        .collect();

    TypedExpr {
        expr: expr.clone(),
        ty,
        children,
    }
}

/// Collect the direct TLExpr sub-expressions of `expr` as a flat `Vec<&TLExpr>`.
fn collect_children(expr: &TLExpr) -> Vec<&TLExpr> {
    match expr {
        TLExpr::And(l, r)
        | TLExpr::Or(l, r)
        | TLExpr::Imply(l, r)
        | TLExpr::Add(l, r)
        | TLExpr::Sub(l, r)
        | TLExpr::Mul(l, r)
        | TLExpr::Div(l, r)
        | TLExpr::Pow(l, r)
        | TLExpr::Mod(l, r)
        | TLExpr::Min(l, r)
        | TLExpr::Max(l, r)
        | TLExpr::Eq(l, r)
        | TLExpr::Lt(l, r)
        | TLExpr::Gt(l, r)
        | TLExpr::Lte(l, r)
        | TLExpr::Gte(l, r) => vec![l.as_ref(), r.as_ref()],

        TLExpr::Not(e)
        | TLExpr::Score(e)
        | TLExpr::Abs(e)
        | TLExpr::Floor(e)
        | TLExpr::Ceil(e)
        | TLExpr::Round(e)
        | TLExpr::Sqrt(e)
        | TLExpr::Exp(e)
        | TLExpr::Log(e)
        | TLExpr::Sin(e)
        | TLExpr::Cos(e)
        | TLExpr::Tan(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e)
        | TLExpr::Box(e)
        | TLExpr::Diamond(e)
        | TLExpr::WeightedRule { rule: e, .. }
        | TLExpr::FuzzyNot { expr: e, .. }
        | TLExpr::LeastFixpoint { body: e, .. }
        | TLExpr::GreatestFixpoint { body: e, .. }
        | TLExpr::Lambda { body: e, .. }
        | TLExpr::SetCardinality { set: e }
        | TLExpr::Somewhere { formula: e }
        | TLExpr::Everywhere { formula: e }
        | TLExpr::Explain { formula: e }
        | TLExpr::At { formula: e, .. } => vec![e.as_ref()],

        TLExpr::ForAll { body, .. }
        | TLExpr::Exists { body, .. }
        | TLExpr::SoftForAll { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::Aggregate { body, .. }
        | TLExpr::CountingExists { body, .. }
        | TLExpr::CountingForAll { body, .. }
        | TLExpr::ExactCount { body, .. }
        | TLExpr::Majority { body, .. }
        | TLExpr::SetComprehension {
            condition: body, ..
        } => vec![body.as_ref()],

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => vec![
            condition.as_ref(),
            then_branch.as_ref(),
            else_branch.as_ref(),
        ],

        TLExpr::Let { value, body, .. } => vec![value.as_ref(), body.as_ref()],

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            vec![left.as_ref(), right.as_ref()]
        }

        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => vec![premise.as_ref(), conclusion.as_ref()],

        TLExpr::Until { before, after }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => vec![before.as_ref(), after.as_ref()],

        TLExpr::SetUnion { left, right }
        | TLExpr::SetIntersection { left, right }
        | TLExpr::SetDifference { left, right } => vec![left.as_ref(), right.as_ref()],

        TLExpr::SetMembership { element, set } => vec![element.as_ref(), set.as_ref()],

        TLExpr::Apply { function, argument } => vec![function.as_ref(), argument.as_ref()],

        TLExpr::ProbabilisticChoice { alternatives } => {
            alternatives.iter().map(|(_, e)| e).collect()
        }

        TLExpr::GlobalCardinality { values, .. } => values.iter().collect(),

        // Leaf nodes (no TLExpr children).
        TLExpr::Constant(_)
        | TLExpr::Pred { .. }
        | TLExpr::EmptySet
        | TLExpr::AllDifferent { .. }
        | TLExpr::Abducible { .. }
        | TLExpr::Nominal { .. }
        | TLExpr::SymbolLiteral(_) => vec![],

        TLExpr::Match { scrutinee, arms } => {
            let mut children = vec![scrutinee.as_ref()];
            children.extend(arms.iter().map(|(_, b)| b.as_ref()));
            children
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TCoNormKind, TLExpr, TNormKind};

    // Helper: build a zero-arity Pred (proposition).
    fn prop(name: &str) -> TLExpr {
        TLExpr::pred(name, vec![])
    }

    // ── 1. Constant infers Numeric ─────────────────────────────────────────
    #[test]
    fn test_constant_is_numeric() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let ty = infer(&TLExpr::Constant(42.0), &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Numeric);
    }

    // ── 2. Zero-arity Pred infers Bool ────────────────────────────────────
    #[test]
    fn test_zero_arity_pred_is_bool() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let ty = infer(&prop("p"), &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Bool);
    }

    // ── 3. Binary Pred infers Relation(2) ─────────────────────────────────
    #[test]
    fn test_binary_pred_is_relation2() {
        use tensorlogic_ir::Term;
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Relation(2));
    }

    // ── 4. And(Bool, Bool) infers Bool ────────────────────────────────────
    #[test]
    fn test_and_bool_bool_is_bool() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::and(prop("p"), prop("q"));
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Bool);
    }

    // ── 5. Add(Numeric, Numeric) infers Numeric ───────────────────────────
    #[test]
    fn test_add_numeric_numeric_is_numeric() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Numeric);
    }

    // ── 6. Not(Bool) infers Bool ──────────────────────────────────────────
    #[test]
    fn test_not_bool_is_bool() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::negate(prop("p"));
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Bool);
    }

    // ── 7. ForAll(var, Bool_body) infers Bool ─────────────────────────────
    #[test]
    fn test_forall_bool_body_is_bool() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::forall("x", "Entity", prop("P"));
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Bool);
    }

    // ── 8. SoftExists infers Fuzzy ────────────────────────────────────────
    #[test]
    fn test_soft_exists_is_fuzzy() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::SoftExists {
            var: "x".into(),
            domain: "D".into(),
            body: Box::new(prop("P")),
            temperature: 1.0,
        };
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Fuzzy);
    }

    // ── 9. TNorm infers Fuzzy ─────────────────────────────────────────────
    #[test]
    fn test_tnorm_is_fuzzy() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::TNorm {
            kind: TNormKind::Product,
            left: Box::new(TLExpr::Constant(0.7)),
            right: Box::new(TLExpr::Constant(0.3)),
        };
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Fuzzy);
    }

    // ── 10. ProbabilisticChoice infers Probabilistic ──────────────────────
    #[test]
    fn test_probabilistic_choice_is_probabilistic() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::ProbabilisticChoice {
            alternatives: vec![(0.6, prop("A")), (0.4, prop("B"))],
        };
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Probabilistic);
    }

    // ── 11. Eq(Numeric, Numeric) infers Bool ──────────────────────────────
    #[test]
    fn test_eq_numeric_is_bool() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::Eq(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(1.0)),
        );
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Bool);
    }

    // ── 12. IfThenElse: condition must be Bool ────────────────────────────
    #[test]
    fn test_ifthenelse_condition_is_bool() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::IfThenElse {
            condition: Box::new(prop("cond")),
            then_branch: Box::new(TLExpr::Constant(1.0)),
            else_branch: Box::new(TLExpr::Constant(0.0)),
        };
        // Should succeed: cond is Bool, branches both Numeric.
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Numeric);
    }

    // ── 12b. IfThenElse with Numeric condition should fail ────────────────
    #[test]
    fn test_ifthenelse_numeric_condition_fails() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::IfThenElse {
            condition: Box::new(TLExpr::Constant(1.0)), // Numeric, not Bool
            then_branch: Box::new(TLExpr::Constant(2.0)),
            else_branch: Box::new(TLExpr::Constant(3.0)),
        };
        let result = infer(&expr, &env, &mut subst, &mut counter);
        assert!(result.is_err(), "expected type error for Numeric condition");
    }

    // ── 13. Let binding extends type environment ──────────────────────────
    #[test]
    fn test_let_binding_extends_env() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        // let x = 1.0 in And(p, q) -- body is Bool (ignores x binding)
        let expr = TLExpr::Let {
            var: "x".into(),
            value: Box::new(TLExpr::Constant(1.0)),
            body: Box::new(TLExpr::and(prop("p"), prop("q"))),
        };
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Bool);
    }

    // ── 14. unify(Bool, Bool) succeeds ────────────────────────────────────
    #[test]
    fn test_unify_bool_bool_ok() {
        let mut subst = Substitution::new();
        assert!(unify(&TLType::Bool, &TLType::Bool, &mut subst).is_ok());
    }

    // ── 15. unify(Bool, Numeric) fails ────────────────────────────────────
    #[test]
    fn test_unify_bool_numeric_fails() {
        let mut subst = Substitution::new();
        let result = unify(&TLType::Bool, &TLType::Numeric, &mut subst);
        assert!(result.is_err());
        match result.unwrap_err() {
            TypeInferError::UnificationFailed { .. } => {}
            e => panic!("expected UnificationFailed, got {:?}", e),
        }
    }

    // ── 16. Unify(Var(0), Bool) → Var(0) resolves to Bool ─────────────────
    #[test]
    fn test_unify_var_resolves() {
        let mut subst = Substitution::new();
        unify(&TLType::Var(0), &TLType::Bool, &mut subst).unwrap();
        let resolved = subst.apply(&TLType::Var(0));
        assert_eq!(resolved, TLType::Bool);
    }

    // ── 17. Occurs check: Var(0) vs Relation(0) doesn't infinite loop ─────
    #[test]
    fn test_occurs_check_no_infinite_loop() {
        // Var(0) does NOT appear inside Relation(0) because Relation(n) has no
        // nested TLType children — the occurs check should succeed (no match).
        let mut subst = Substitution::new();
        let result = unify(&TLType::Var(0), &TLType::Relation(0), &mut subst);
        assert!(result.is_ok(), "Var(0) vs Relation(0) should unify fine");
        let resolved = subst.apply(&TLType::Var(0));
        assert_eq!(resolved, TLType::Relation(0));
    }

    // ── 18. infer_type returns TypeInferResult with correct type ──────────
    #[test]
    fn test_infer_type_result() {
        let expr = TLExpr::and(prop("p"), prop("q"));
        let result = infer_type(&expr, &TypeEnv::new());
        assert_eq!(result.typed_expr.ty, TLType::Bool);
        assert!(result.errors.is_empty());
    }

    // ── 19. annotate returns TypedExpr with type at root ──────────────────
    #[test]
    fn test_annotate_root_type() {
        let expr = TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        let typed = annotate(&expr, &TypeEnv::new()).unwrap();
        assert_eq!(typed.ty, TLType::Numeric);
    }

    // ── 20. TLType::is_ground() returns false for Var(_) ──────────────────
    #[test]
    fn test_is_ground_false_for_var() {
        assert!(!TLType::Var(99).is_ground());
        assert!(TLType::Bool.is_ground());
        assert!(TLType::Numeric.is_ground());
        assert!(TLType::Relation(3).is_ground());
    }

    // ── 21. TypeEnv::extend creates new env without mutating original ──────
    #[test]
    fn test_type_env_extend_non_destructive() {
        let env = TypeEnv::new().with("x", TLType::Bool);
        let extended = env.extend("y", TLType::Numeric);
        // Original has x but not y.
        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_none());
        // Extended has both.
        assert!(extended.lookup("x").is_some());
        assert!(extended.lookup("y").is_some());
    }

    // ── 22. Substitution::apply chases through chains ──────────────────────
    #[test]
    fn test_substitution_apply_chases_chain() {
        let mut subst = Substitution::new();
        subst.bind(0, TLType::Var(1));
        subst.bind(1, TLType::Var(2));
        subst.bind(2, TLType::Numeric);
        let resolved = subst.apply(&TLType::Var(0));
        assert_eq!(resolved, TLType::Numeric);
    }

    // ── 23. infer_type on nested And(Or(p,q), Not(r)) returns Bool ─────────
    #[test]
    fn test_nested_and_or_not_is_bool() {
        let expr = TLExpr::and(TLExpr::or(prop("p"), prop("q")), TLExpr::negate(prop("r")));
        let result = infer_type(&expr, &TypeEnv::new());
        assert_eq!(result.typed_expr.ty, TLType::Bool);
        assert!(result.errors.is_empty());
    }

    // ── 24. TCoNorm infers Fuzzy ──────────────────────────────────────────
    #[test]
    fn test_tconorm_is_fuzzy() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::TCoNorm {
            kind: TCoNormKind::Maximum,
            left: Box::new(TLExpr::Constant(0.2)),
            right: Box::new(TLExpr::Constant(0.8)),
        };
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Fuzzy);
    }

    // ── 25. WeightedRule infers Probabilistic ─────────────────────────────
    #[test]
    fn test_weighted_rule_is_probabilistic() {
        let env = TypeEnv::new();
        let mut subst = Substitution::new();
        let mut counter = TyVarCounter::new();
        let expr = TLExpr::WeightedRule {
            weight: 0.9,
            rule: Box::new(TLExpr::imply(prop("A"), prop("B"))),
        };
        let ty = infer(&expr, &env, &mut subst, &mut counter).unwrap();
        assert_eq!(ty, TLType::Probabilistic);
    }
}
