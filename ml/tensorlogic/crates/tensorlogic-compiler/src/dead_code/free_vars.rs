//! Free-variable analysis used to decide whether Let bindings are dead.

use tensorlogic_ir::TLExpr;

use super::types::DeadCodeEliminator;

impl DeadCodeEliminator {
    /// Returns `true` if `var` appears free (as a predicate argument or
    /// inside a binder that does *not* shadow it) anywhere in `expr`.
    ///
    /// Variable occurrences in TLExpr live in two places:
    /// 1. `Term::Var(name)` inside `Pred` args
    /// 2. `AllDifferent { variables }` variable name lists
    /// 3. Any binder (`Exists`, `ForAll`, `Let`, `Lambda`, `LeastFixpoint`, etc.)
    ///    that introduces a new scope — we stop counting the variable as free
    ///    inside the scope if the binder's name equals `var`.
    pub fn is_free(&self, var: &str, expr: &TLExpr) -> bool {
        self.is_free_in(var, expr, false)
    }

    /// Internal helper for `is_free`, carrying a `shadowed` flag.
    pub(super) fn is_free_in(&self, var: &str, expr: &TLExpr, shadowed: bool) -> bool {
        if shadowed {
            return false;
        }
        match expr {
            TLExpr::Pred { args, .. } => args
                .iter()
                .any(|t| matches!(t, tensorlogic_ir::Term::Var(v) if v == var)),

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
            | TLExpr::Gte(l, r) => self.is_free_in(var, l, false) || self.is_free_in(var, r, false),

            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                self.is_free_in(var, left, false) || self.is_free_in(var, right, false)
            }

            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => self.is_free_in(var, premise, false) || self.is_free_in(var, conclusion, false),

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
            | TLExpr::Box(e)
            | TLExpr::Diamond(e)
            | TLExpr::Next(e)
            | TLExpr::Eventually(e)
            | TLExpr::Always(e) => self.is_free_in(var, e, false),

            TLExpr::FuzzyNot { expr, .. } => self.is_free_in(var, expr, false),

            TLExpr::WeightedRule { rule, .. } => self.is_free_in(var, rule, false),

            TLExpr::Until { before, after }
            | TLExpr::Release {
                released: before,
                releaser: after,
            }
            | TLExpr::WeakUntil { before, after }
            | TLExpr::StrongRelease {
                released: before,
                releaser: after,
            } => self.is_free_in(var, before, false) || self.is_free_in(var, after, false),

            TLExpr::Exists {
                var: binder, body, ..
            }
            | TLExpr::ForAll {
                var: binder, body, ..
            }
            | TLExpr::SoftExists {
                var: binder, body, ..
            }
            | TLExpr::SoftForAll {
                var: binder, body, ..
            } => self.is_free_in(var, body, binder == var),

            TLExpr::Aggregate {
                var: binder,
                body,
                group_by,
                ..
            } => {
                let in_body = self.is_free_in(var, body, binder == var);
                let in_group = group_by
                    .as_ref()
                    .map(|gs| gs.iter().any(|g| g == var))
                    .unwrap_or(false);
                in_body || in_group
            }

            TLExpr::Let {
                var: binder,
                value,
                body,
            } => self.is_free_in(var, value, false) || self.is_free_in(var, body, binder == var),

            TLExpr::Lambda {
                var: binder, body, ..
            } => self.is_free_in(var, body, binder == var),

            TLExpr::Apply { function, argument } => {
                self.is_free_in(var, function, false) || self.is_free_in(var, argument, false)
            }

            TLExpr::SetMembership { element, set }
            | TLExpr::SetUnion {
                left: element,
                right: set,
            }
            | TLExpr::SetIntersection {
                left: element,
                right: set,
            }
            | TLExpr::SetDifference {
                left: element,
                right: set,
            } => self.is_free_in(var, element, false) || self.is_free_in(var, set, false),

            TLExpr::SetCardinality { set } => self.is_free_in(var, set, false),

            TLExpr::SetComprehension {
                var: binder,
                condition,
                ..
            } => self.is_free_in(var, condition, binder == var),

            TLExpr::CountingExists {
                var: binder, body, ..
            }
            | TLExpr::CountingForAll {
                var: binder, body, ..
            }
            | TLExpr::ExactCount {
                var: binder, body, ..
            }
            | TLExpr::Majority {
                var: binder, body, ..
            } => self.is_free_in(var, body, binder == var),

            TLExpr::LeastFixpoint { var: binder, body }
            | TLExpr::GreatestFixpoint { var: binder, body } => {
                self.is_free_in(var, body, binder == var)
            }

            TLExpr::At { formula, .. } => self.is_free_in(var, formula, false),
            TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
                self.is_free_in(var, formula, false)
            }

            TLExpr::Explain { formula } => self.is_free_in(var, formula, false),

            TLExpr::AllDifferent { variables } => variables.iter().any(|v| v == var),

            TLExpr::GlobalCardinality {
                variables, values, ..
            } => {
                variables.iter().any(|v| v == var)
                    || values.iter().any(|e| self.is_free_in(var, e, false))
            }

            TLExpr::ProbabilisticChoice { alternatives } => alternatives
                .iter()
                .any(|(_, e)| self.is_free_in(var, e, false)),

            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.is_free_in(var, condition, false)
                    || self.is_free_in(var, then_branch, false)
                    || self.is_free_in(var, else_branch, false)
            }

            TLExpr::Constant(_)
            | TLExpr::EmptySet
            | TLExpr::Nominal { .. }
            | TLExpr::Abducible { .. }
            | TLExpr::SymbolLiteral(_) => false,

            TLExpr::Match { scrutinee, arms } => {
                self.is_free_in(var, scrutinee, false)
                    || arms.iter().any(|(_, b)| self.is_free_in(var, b, false))
            }
        }
    }
}
