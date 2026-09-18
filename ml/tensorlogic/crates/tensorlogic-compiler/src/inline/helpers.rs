use tensorlogic_ir::TLExpr;

/// Count how many times `var` appears free in `expr`.
///
/// A variable occurrence is:
/// - A `Pred { name, args: [] }` node whose `name == var` (zero-arg
///   predicate used as a variable reference in let bodies), OR
/// - A `Term::Var(v)` inside `Pred` args where `v == var`.
///
/// The count respects capture: once a binder with the same name is
/// entered, occurrences inside that scope are not counted as free.
pub fn count_free_occurrences(var: &str, expr: &TLExpr) -> usize {
    count_free_in(var, expr)
}

pub(crate) fn count_free_in(var: &str, expr: &TLExpr) -> usize {
    match expr {
        // A zero-argument predicate serves as a variable reference.
        TLExpr::Pred { name, args } => {
            if args.is_empty() && name == var {
                1
            } else {
                // Count Term::Var occurrences in the argument list.
                args.iter()
                    .filter(|t| matches!(t, tensorlogic_ir::Term::Var(v) if v == var))
                    .count()
            }
        }

        // ── Binary nodes ─────────────────────────────────────────────────
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
        | TLExpr::Gte(l, r) => count_free_in(var, l) + count_free_in(var, r),

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            count_free_in(var, left) + count_free_in(var, right)
        }
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => count_free_in(var, premise) + count_free_in(var, conclusion),

        // ── Unary nodes ──────────────────────────────────────────────────
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
        | TLExpr::Always(e) => count_free_in(var, e),

        TLExpr::FuzzyNot { expr, .. } => count_free_in(var, expr),
        TLExpr::WeightedRule { rule, .. } => count_free_in(var, rule),

        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => count_free_in(var, before) + count_free_in(var, after),

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            count_free_in(var, condition)
                + count_free_in(var, then_branch)
                + count_free_in(var, else_branch)
        }

        TLExpr::Apply { function, argument } => {
            count_free_in(var, function) + count_free_in(var, argument)
        }

        // ── Binders — shadow the variable in the body ────────────────────
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
        }
        | TLExpr::CountingExists {
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
        }
        | TLExpr::LeastFixpoint { var: binder, body }
        | TLExpr::GreatestFixpoint { var: binder, body } => {
            if binder == var {
                0
            } else {
                count_free_in(var, body)
            }
        }

        TLExpr::Lambda {
            var: binder, body, ..
        } => {
            if binder == var {
                0
            } else {
                count_free_in(var, body)
            }
        }

        TLExpr::Aggregate {
            var: binder,
            body,
            group_by,
            ..
        } => {
            let in_body = if binder == var {
                0
            } else {
                count_free_in(var, body)
            };
            let in_group = group_by
                .as_ref()
                .map(|gs| gs.iter().filter(|g| g.as_str() == var).count())
                .unwrap_or(0);
            in_body + in_group
        }

        // For Let: value is in scope of outer env, body is in scope of
        // the binding; if binder == var, occurrences in body are shadowed.
        TLExpr::Let {
            var: binder,
            value,
            body,
        } => {
            let in_value = count_free_in(var, value);
            let in_body = if binder == var {
                0
            } else {
                count_free_in(var, body)
            };
            in_value + in_body
        }

        TLExpr::SetComprehension {
            var: binder,
            condition,
            ..
        } => {
            if binder == var {
                0
            } else {
                count_free_in(var, condition)
            }
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
        } => count_free_in(var, element) + count_free_in(var, set),

        TLExpr::SetCardinality { set } => count_free_in(var, set),

        TLExpr::At { formula, .. } => count_free_in(var, formula),
        TLExpr::Somewhere { formula } | TLExpr::Everywhere { formula } => {
            count_free_in(var, formula)
        }
        TLExpr::Explain { formula } => count_free_in(var, formula),

        TLExpr::ProbabilisticChoice { alternatives } => alternatives
            .iter()
            .map(|(_, e)| count_free_in(var, e))
            .sum(),

        TLExpr::AllDifferent { variables } => {
            variables.iter().filter(|v| v.as_str() == var).count()
        }
        TLExpr::GlobalCardinality {
            variables, values, ..
        } => {
            let in_vars = variables.iter().filter(|v| v.as_str() == var).count();
            let in_vals: usize = values.iter().map(|e| count_free_in(var, e)).sum();
            in_vars + in_vals
        }

        // ── Leaves with no variable occurrences ──────────────────────────
        TLExpr::Constant(_)
        | TLExpr::EmptySet
        | TLExpr::Nominal { .. }
        | TLExpr::Abducible { .. }
        | TLExpr::SymbolLiteral(_) => 0,

        TLExpr::Match { scrutinee, arms } => {
            count_free_in(var, scrutinee)
                + arms
                    .iter()
                    .map(|(_, b)| count_free_in(var, b))
                    .sum::<usize>()
        }
    }
}

/// Returns `true` if `expr` is a constant literal (`Constant(_)`).
pub fn is_constant_binding(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Constant(_))
}

/// Returns `true` if `expr` is a zero-argument predicate (variable alias).
pub fn is_var_binding(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Pred { args, .. } if args.is_empty())
}

/// Returns `true` if `expr` is a "simple" binding worth inlining regardless
/// of use count: either a constant or a variable alias.
pub fn is_simple_binding(expr: &TLExpr) -> bool {
    is_constant_binding(expr) || is_var_binding(expr)
}

/// Compute the depth (height) of an expression tree.
///
/// Leaf nodes have depth 1; each internal node adds 1 to the maximum
/// depth of its children.
pub fn expr_depth(expr: &TLExpr) -> usize {
    match expr {
        // ── Binary nodes ─────────────────────────────────────────────────
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
        | TLExpr::Gte(l, r) => 1 + expr_depth(l).max(expr_depth(r)),

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            1 + expr_depth(left).max(expr_depth(right))
        }
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => 1 + expr_depth(premise).max(expr_depth(conclusion)),

        // ── Unary nodes ──────────────────────────────────────────────────
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
        | TLExpr::Always(e) => 1 + expr_depth(e),

        TLExpr::FuzzyNot { expr, .. } => 1 + expr_depth(expr),
        TLExpr::WeightedRule { rule, .. } => 1 + expr_depth(rule),

        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => 1 + expr_depth(before).max(expr_depth(after)),

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            1 + expr_depth(condition)
                .max(expr_depth(then_branch))
                .max(expr_depth(else_branch))
        }

        TLExpr::Exists { body, .. }
        | TLExpr::ForAll { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::SoftForAll { body, .. }
        | TLExpr::Aggregate { body, .. }
        | TLExpr::Lambda { body, .. }
        | TLExpr::SetComprehension {
            condition: body, ..
        }
        | TLExpr::CountingExists { body, .. }
        | TLExpr::CountingForAll { body, .. }
        | TLExpr::ExactCount { body, .. }
        | TLExpr::Majority { body, .. }
        | TLExpr::LeastFixpoint { body, .. }
        | TLExpr::GreatestFixpoint { body, .. } => 1 + expr_depth(body),

        TLExpr::Let { value, body, .. } => 1 + expr_depth(value).max(expr_depth(body)),

        TLExpr::Apply { function, argument } => 1 + expr_depth(function).max(expr_depth(argument)),

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
        } => 1 + expr_depth(element).max(expr_depth(set)),

        TLExpr::SetCardinality { set } => 1 + expr_depth(set),

        TLExpr::At { formula, .. } => 1 + expr_depth(formula),
        TLExpr::Somewhere { formula }
        | TLExpr::Everywhere { formula }
        | TLExpr::Explain { formula } => 1 + expr_depth(formula),

        TLExpr::ProbabilisticChoice { alternatives } => {
            let max_depth = alternatives
                .iter()
                .map(|(_, e)| expr_depth(e))
                .max()
                .unwrap_or(0);
            1 + max_depth
        }

        // ── Leaves ───────────────────────────────────────────────────────
        TLExpr::Pred { .. }
        | TLExpr::Constant(_)
        | TLExpr::EmptySet
        | TLExpr::AllDifferent { .. }
        | TLExpr::GlobalCardinality { .. }
        | TLExpr::Nominal { .. }
        | TLExpr::Abducible { .. }
        | TLExpr::SymbolLiteral(_) => 1,

        TLExpr::Match { scrutinee, arms } => {
            1 + expr_depth(scrutinee) + arms.iter().map(|(_, b)| expr_depth(b)).max().unwrap_or(0)
        }
    }
}

/// Count total nodes in an expression tree.
pub fn count_nodes(expr: &TLExpr) -> u64 {
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
        | TLExpr::Gte(l, r) => 1 + count_nodes(l) + count_nodes(r),

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            1 + count_nodes(left) + count_nodes(right)
        }
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => 1 + count_nodes(premise) + count_nodes(conclusion),

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
        | TLExpr::Always(e) => 1 + count_nodes(e),

        TLExpr::FuzzyNot { expr, .. } => 1 + count_nodes(expr),
        TLExpr::WeightedRule { rule, .. } => 1 + count_nodes(rule),

        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => 1 + count_nodes(before) + count_nodes(after),

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => 1 + count_nodes(condition) + count_nodes(then_branch) + count_nodes(else_branch),

        TLExpr::Exists { body, .. }
        | TLExpr::ForAll { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::SoftForAll { body, .. }
        | TLExpr::Aggregate { body, .. }
        | TLExpr::Lambda { body, .. }
        | TLExpr::SetComprehension {
            condition: body, ..
        }
        | TLExpr::CountingExists { body, .. }
        | TLExpr::CountingForAll { body, .. }
        | TLExpr::ExactCount { body, .. }
        | TLExpr::Majority { body, .. }
        | TLExpr::LeastFixpoint { body, .. }
        | TLExpr::GreatestFixpoint { body, .. } => 1 + count_nodes(body),

        TLExpr::Let { value, body, .. } => 1 + count_nodes(value) + count_nodes(body),

        TLExpr::Apply { function, argument } => 1 + count_nodes(function) + count_nodes(argument),

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
        } => 1 + count_nodes(element) + count_nodes(set),

        TLExpr::SetCardinality { set } => 1 + count_nodes(set),

        TLExpr::At { formula, .. } => 1 + count_nodes(formula),
        TLExpr::Somewhere { formula }
        | TLExpr::Everywhere { formula }
        | TLExpr::Explain { formula } => 1 + count_nodes(formula),

        TLExpr::ProbabilisticChoice { alternatives } => {
            1 + alternatives
                .iter()
                .map(|(_, e)| count_nodes(e))
                .sum::<u64>()
        }

        TLExpr::Pred { .. }
        | TLExpr::Constant(_)
        | TLExpr::EmptySet
        | TLExpr::AllDifferent { .. }
        | TLExpr::GlobalCardinality { .. }
        | TLExpr::Nominal { .. }
        | TLExpr::Abducible { .. }
        | TLExpr::SymbolLiteral(_) => 1,

        TLExpr::Match { scrutinee, arms } => {
            1 + count_nodes(scrutinee) + arms.iter().map(|(_, b)| count_nodes(b)).sum::<u64>()
        }
    }
}
