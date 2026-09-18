//! Expression canonicalization for improved cache hit rates.
//!
//! Puts TLExpr into a canonical structural form so that semantically
//! equivalent expressions (e.g., AND(a,b) vs AND(b,a)) have identical
//! fingerprints. This is purely structural normalization — not constant
//! folding or algebraic simplification (those exist in separate passes).
//!
//! Three canonicalization rules:
//! 1. **Double negation elimination**: NOT(NOT(x)) → x
//! 2. **Nested same-op flattening**: AND(AND(a,b), c) → AND(a, AND(b,c)) sorted
//! 3. **Commutative sorting**: AND/OR operands sorted by canonical_order_key

use tensorlogic_ir::TLExpr;

/// Statistics from canonicalization.
#[derive(Debug, Clone, Default)]
pub struct CanonicalStats {
    /// Number of double negations removed
    pub double_neg_removed: usize,
    /// Number of commutative sorts applied
    pub commutative_sorted: usize,
    /// Number of nested same-op flattened
    pub nested_flattened: usize,
    /// Total rewrites performed
    pub total_rewrites: usize,
}

impl CanonicalStats {
    /// Merge another stats into this one.
    pub fn merge(&mut self, other: &CanonicalStats) {
        self.double_neg_removed += other.double_neg_removed;
        self.commutative_sorted += other.commutative_sorted;
        self.nested_flattened += other.nested_flattened;
        self.total_rewrites += other.total_rewrites;
    }
}

impl std::fmt::Display for CanonicalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CanonicalStats {{ double_neg: {}, comm_sorted: {}, flattened: {}, total: {} }}",
            self.double_neg_removed,
            self.commutative_sorted,
            self.nested_flattened,
            self.total_rewrites
        )
    }
}

/// Expression canonicalizer with configurable rules.
#[derive(Debug, Clone)]
pub struct Canonicalizer {
    /// Whether to sort commutative operands (AND, OR).
    pub sort_commutative: bool,
    /// Whether to flatten nested same-op expressions.
    pub flatten_nested: bool,
    /// Whether to eliminate double negations.
    pub elim_double_neg: bool,
}

impl Default for Canonicalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Canonicalizer {
    /// Create a new canonicalizer with all rules enabled.
    pub fn new() -> Self {
        Canonicalizer {
            sort_commutative: true,
            flatten_nested: true,
            elim_double_neg: true,
        }
    }

    /// Set whether to sort commutative operands.
    pub fn with_sort_commutative(mut self, v: bool) -> Self {
        self.sort_commutative = v;
        self
    }

    /// Set whether to flatten nested same-op expressions.
    pub fn with_flatten_nested(mut self, v: bool) -> Self {
        self.flatten_nested = v;
        self
    }

    /// Set whether to eliminate double negations.
    pub fn with_elim_double_neg(mut self, v: bool) -> Self {
        self.elim_double_neg = v;
        self
    }

    /// Canonicalize an expression, returning the normalized form and stats.
    pub fn canonicalize(&self, expr: &TLExpr) -> (TLExpr, CanonicalStats) {
        let mut stats = CanonicalStats::default();
        let result = self.normalize(expr, &mut stats);
        (result, stats)
    }

    /// Produce a deterministic canonical key string for cache use.
    pub fn canonical_key(&self, expr: &TLExpr) -> String {
        let (normalized, _) = self.canonicalize(expr);
        format!("{:?}", normalized)
    }

    fn normalize(&self, expr: &TLExpr, stats: &mut CanonicalStats) -> TLExpr {
        match expr {
            // Double negation elimination: NOT(NOT(x)) → x
            TLExpr::Not(inner) => {
                if self.elim_double_neg {
                    if let TLExpr::Not(inner_inner) = inner.as_ref() {
                        stats.double_neg_removed += 1;
                        stats.total_rewrites += 1;
                        return self.normalize(inner_inner, stats);
                    }
                }
                TLExpr::negate(self.normalize(inner, stats))
            }

            // AND: flatten nested AND, then sort commutative operands
            TLExpr::And(a, b) => {
                let norm_a = self.normalize(a, stats);
                let norm_b = self.normalize(b, stats);
                let mut operands = Vec::new();
                if self.flatten_nested {
                    self.collect_and_operands(&norm_a, &mut operands, stats);
                    self.collect_and_operands(&norm_b, &mut operands, stats);
                } else {
                    operands.push(norm_a);
                    operands.push(norm_b);
                }
                if self.sort_commutative {
                    let before = operands.iter().map(canonical_order_key).collect::<Vec<_>>();
                    operands.sort_by_key(canonical_order_key);
                    let after = operands.iter().map(canonical_order_key).collect::<Vec<_>>();
                    if before != after {
                        stats.commutative_sorted += 1;
                        stats.total_rewrites += 1;
                    }
                }
                self.build_right_leaning_and(operands)
            }

            // OR: flatten nested OR, then sort commutative operands
            TLExpr::Or(a, b) => {
                let norm_a = self.normalize(a, stats);
                let norm_b = self.normalize(b, stats);
                let mut operands = Vec::new();
                if self.flatten_nested {
                    self.collect_or_operands(&norm_a, &mut operands, stats);
                    self.collect_or_operands(&norm_b, &mut operands, stats);
                } else {
                    operands.push(norm_a);
                    operands.push(norm_b);
                }
                if self.sort_commutative {
                    let before = operands.iter().map(canonical_order_key).collect::<Vec<_>>();
                    operands.sort_by_key(canonical_order_key);
                    let after = operands.iter().map(canonical_order_key).collect::<Vec<_>>();
                    if before != after {
                        stats.commutative_sorted += 1;
                        stats.total_rewrites += 1;
                    }
                }
                self.build_right_leaning_or(operands)
            }

            // Recurse into binary operators (non-commutative in canonicalization sense)
            TLExpr::Imply(a, b) => {
                TLExpr::imply(self.normalize(a, stats), self.normalize(b, stats))
            }
            TLExpr::Add(a, b) => TLExpr::add(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Sub(a, b) => TLExpr::sub(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Mul(a, b) => TLExpr::mul(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Div(a, b) => TLExpr::div(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Pow(a, b) => TLExpr::pow(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Mod(a, b) => TLExpr::modulo(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Min(a, b) => TLExpr::min(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Max(a, b) => TLExpr::max(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Eq(a, b) => TLExpr::eq(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Lt(a, b) => TLExpr::lt(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Gt(a, b) => TLExpr::gt(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Lte(a, b) => TLExpr::lte(self.normalize(a, stats), self.normalize(b, stats)),
            TLExpr::Gte(a, b) => TLExpr::gte(self.normalize(a, stats), self.normalize(b, stats)),

            // Unary math
            TLExpr::Abs(inner) => TLExpr::abs(self.normalize(inner, stats)),
            TLExpr::Floor(inner) => TLExpr::floor(self.normalize(inner, stats)),
            TLExpr::Ceil(inner) => TLExpr::ceil(self.normalize(inner, stats)),
            TLExpr::Round(inner) => TLExpr::round(self.normalize(inner, stats)),
            TLExpr::Sqrt(inner) => TLExpr::sqrt(self.normalize(inner, stats)),
            TLExpr::Exp(inner) => TLExpr::exp(self.normalize(inner, stats)),
            TLExpr::Log(inner) => TLExpr::log(self.normalize(inner, stats)),
            TLExpr::Sin(inner) => TLExpr::sin(self.normalize(inner, stats)),
            TLExpr::Cos(inner) => TLExpr::cos(self.normalize(inner, stats)),
            TLExpr::Tan(inner) => TLExpr::tan(self.normalize(inner, stats)),
            TLExpr::Score(inner) => TLExpr::score(self.normalize(inner, stats)),

            // Quantifiers
            TLExpr::Exists { var, domain, body } => {
                TLExpr::exists(var.clone(), domain.clone(), self.normalize(body, stats))
            }
            TLExpr::ForAll { var, domain, body } => {
                TLExpr::forall(var.clone(), domain.clone(), self.normalize(body, stats))
            }

            // Conditional
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => TLExpr::if_then_else(
                self.normalize(condition, stats),
                self.normalize(then_branch, stats),
                self.normalize(else_branch, stats),
            ),

            // Let binding
            TLExpr::Let { var, value, body } => TLExpr::let_binding(
                var.clone(),
                self.normalize(value, stats),
                self.normalize(body, stats),
            ),

            // Aggregate
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body,
                group_by,
            } => TLExpr::Aggregate {
                op: op.clone(),
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
                group_by: group_by.clone(),
            },

            // Fuzzy logic operators
            TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
                kind: *kind,
                left: Box::new(self.normalize(left, stats)),
                right: Box::new(self.normalize(right, stats)),
            },
            TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
                kind: *kind,
                left: Box::new(self.normalize(left, stats)),
                right: Box::new(self.normalize(right, stats)),
            },
            TLExpr::FuzzyNot { kind, expr: inner } => TLExpr::FuzzyNot {
                kind: *kind,
                expr: Box::new(self.normalize(inner, stats)),
            },
            TLExpr::FuzzyImplication {
                kind,
                premise,
                conclusion,
            } => TLExpr::FuzzyImplication {
                kind: *kind,
                premise: Box::new(self.normalize(premise, stats)),
                conclusion: Box::new(self.normalize(conclusion, stats)),
            },

            // Soft quantifiers
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => TLExpr::SoftExists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
                temperature: *temperature,
            },
            TLExpr::SoftForAll {
                var,
                domain,
                body,
                temperature,
            } => TLExpr::SoftForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
                temperature: *temperature,
            },

            // Weighted rule
            TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
                weight: *weight,
                rule: Box::new(self.normalize(rule, stats)),
            },

            // Probabilistic choice
            TLExpr::ProbabilisticChoice { alternatives } => {
                let norm_alts: Vec<_> = alternatives
                    .iter()
                    .map(|(w, e)| (*w, self.normalize(e, stats)))
                    .collect();
                TLExpr::ProbabilisticChoice {
                    alternatives: norm_alts,
                }
            }

            // Modal logic
            TLExpr::Box(inner) => TLExpr::Box(Box::new(self.normalize(inner, stats))),
            TLExpr::Diamond(inner) => TLExpr::Diamond(Box::new(self.normalize(inner, stats))),

            // Temporal logic
            TLExpr::Next(inner) => TLExpr::Next(Box::new(self.normalize(inner, stats))),
            TLExpr::Eventually(inner) => TLExpr::Eventually(Box::new(self.normalize(inner, stats))),
            TLExpr::Always(inner) => TLExpr::Always(Box::new(self.normalize(inner, stats))),
            TLExpr::Until { before, after } => TLExpr::Until {
                before: Box::new(self.normalize(before, stats)),
                after: Box::new(self.normalize(after, stats)),
            },
            TLExpr::Release { released, releaser } => TLExpr::Release {
                released: Box::new(self.normalize(released, stats)),
                releaser: Box::new(self.normalize(releaser, stats)),
            },
            TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
                before: Box::new(self.normalize(before, stats)),
                after: Box::new(self.normalize(after, stats)),
            },
            TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
                released: Box::new(self.normalize(released, stats)),
                releaser: Box::new(self.normalize(releaser, stats)),
            },

            // Higher-order
            TLExpr::Lambda {
                var,
                var_type,
                body,
            } => TLExpr::Lambda {
                var: var.clone(),
                var_type: var_type.clone(),
                body: Box::new(self.normalize(body, stats)),
            },
            TLExpr::Apply { function, argument } => TLExpr::Apply {
                function: Box::new(self.normalize(function, stats)),
                argument: Box::new(self.normalize(argument, stats)),
            },

            // Set operations
            TLExpr::SetMembership { element, set } => TLExpr::SetMembership {
                element: Box::new(self.normalize(element, stats)),
                set: Box::new(self.normalize(set, stats)),
            },
            TLExpr::SetUnion { left, right } => TLExpr::SetUnion {
                left: Box::new(self.normalize(left, stats)),
                right: Box::new(self.normalize(right, stats)),
            },
            TLExpr::SetIntersection { left, right } => TLExpr::SetIntersection {
                left: Box::new(self.normalize(left, stats)),
                right: Box::new(self.normalize(right, stats)),
            },
            TLExpr::SetDifference { left, right } => TLExpr::SetDifference {
                left: Box::new(self.normalize(left, stats)),
                right: Box::new(self.normalize(right, stats)),
            },
            TLExpr::SetCardinality { set } => TLExpr::SetCardinality {
                set: Box::new(self.normalize(set, stats)),
            },
            TLExpr::SetComprehension {
                var,
                domain,
                condition,
            } => TLExpr::SetComprehension {
                var: var.clone(),
                domain: domain.clone(),
                condition: Box::new(self.normalize(condition, stats)),
            },

            // Counting quantifiers
            TLExpr::CountingExists {
                var,
                domain,
                body,
                min_count,
            } => TLExpr::CountingExists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
                min_count: *min_count,
            },
            TLExpr::CountingForAll {
                var,
                domain,
                body,
                min_count,
            } => TLExpr::CountingForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
                min_count: *min_count,
            },
            TLExpr::ExactCount {
                var,
                domain,
                body,
                count,
            } => TLExpr::ExactCount {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
                count: *count,
            },
            TLExpr::Majority { var, domain, body } => TLExpr::Majority {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(self.normalize(body, stats)),
            },

            // Fixed-point operators
            TLExpr::LeastFixpoint { var, body } => TLExpr::LeastFixpoint {
                var: var.clone(),
                body: Box::new(self.normalize(body, stats)),
            },
            TLExpr::GreatestFixpoint { var, body } => TLExpr::GreatestFixpoint {
                var: var.clone(),
                body: Box::new(self.normalize(body, stats)),
            },

            // Hybrid logic
            TLExpr::At { nominal, formula } => TLExpr::At {
                nominal: nominal.clone(),
                formula: Box::new(self.normalize(formula, stats)),
            },
            TLExpr::Somewhere { formula } => TLExpr::Somewhere {
                formula: Box::new(self.normalize(formula, stats)),
            },
            TLExpr::Everywhere { formula } => TLExpr::Everywhere {
                formula: Box::new(self.normalize(formula, stats)),
            },
            TLExpr::Explain { formula } => TLExpr::Explain {
                formula: Box::new(self.normalize(formula, stats)),
            },

            // Leaves and remaining variants
            TLExpr::Pred { .. }
            | TLExpr::Constant(_)
            | TLExpr::EmptySet
            | TLExpr::Nominal { .. }
            | TLExpr::AllDifferent { .. }
            | TLExpr::GlobalCardinality { .. }
            | TLExpr::Abducible { .. }
            | TLExpr::SymbolLiteral(_) => expr.clone(),

            TLExpr::Match { scrutinee, arms } => TLExpr::Match {
                scrutinee: Box::new(self.normalize(scrutinee, stats)),
                arms: arms
                    .iter()
                    .map(|(p, b)| (p.clone(), Box::new(self.normalize(b, stats))))
                    .collect(),
            },
        }
    }

    /// Collect all operands from nested AND expressions (flattening).
    fn collect_and_operands(
        &self,
        expr: &TLExpr,
        operands: &mut Vec<TLExpr>,
        stats: &mut CanonicalStats,
    ) {
        if let TLExpr::And(a, b) = expr {
            stats.nested_flattened += 1;
            stats.total_rewrites += 1;
            self.collect_and_operands(a, operands, stats);
            self.collect_and_operands(b, operands, stats);
        } else {
            operands.push(expr.clone());
        }
    }

    /// Collect all operands from nested OR expressions (flattening).
    fn collect_or_operands(
        &self,
        expr: &TLExpr,
        operands: &mut Vec<TLExpr>,
        stats: &mut CanonicalStats,
    ) {
        if let TLExpr::Or(a, b) = expr {
            stats.nested_flattened += 1;
            stats.total_rewrites += 1;
            self.collect_or_operands(a, operands, stats);
            self.collect_or_operands(b, operands, stats);
        } else {
            operands.push(expr.clone());
        }
    }

    /// Build a right-leaning AND tree from a list of operands.
    fn build_right_leaning_and(&self, mut operands: Vec<TLExpr>) -> TLExpr {
        match operands.len() {
            0 => TLExpr::Constant(1.0), // identity for AND (true)
            1 => operands.remove(0),
            _ => {
                // Build right-leaning: AND(a, AND(b, AND(c, d)))
                let last = operands.pop();
                operands.into_iter().rev().fold(
                    // Safe: len >= 2 so pop always returns Some
                    last.unwrap_or(TLExpr::Constant(1.0)),
                    |acc, elem| TLExpr::and(elem, acc),
                )
            }
        }
    }

    /// Build a right-leaning OR tree from a list of operands.
    fn build_right_leaning_or(&self, mut operands: Vec<TLExpr>) -> TLExpr {
        match operands.len() {
            0 => TLExpr::Constant(0.0), // identity for OR (false)
            1 => operands.remove(0),
            _ => {
                let last = operands.pop();
                operands
                    .into_iter()
                    .rev()
                    .fold(last.unwrap_or(TLExpr::Constant(0.0)), |acc, elem| {
                        TLExpr::or(elem, acc)
                    })
            }
        }
    }
}

/// Compute a canonical ordering key for sorting commutative children.
///
/// Produces a deterministic string representation suitable for ordering.
/// This ensures AND(a,b) and AND(b,a) sort to the same canonical form.
pub fn canonical_order_key(expr: &TLExpr) -> String {
    match expr {
        TLExpr::Pred { name, args } => format!("P:{}:{}", name, args.len()),
        TLExpr::Constant(v) => {
            // Use a canonical float representation
            if v.is_nan() {
                "C:NaN".to_string()
            } else {
                format!("C:{}", v)
            }
        }
        TLExpr::Not(inner) => format!("Op:Not({})", canonical_order_key(inner)),
        TLExpr::And(a, b) => format!(
            "Op:And({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Or(a, b) => format!(
            "Op:Or({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Imply(a, b) => format!(
            "Op:Imply({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Exists { var, domain, body } => {
            format!("Q:Exists({},{},{})", var, domain, canonical_order_key(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            format!("Q:ForAll({},{},{})", var, domain, canonical_order_key(body))
        }
        TLExpr::Score(inner) => format!("Op:Score({})", canonical_order_key(inner)),
        TLExpr::Add(a, b) => format!(
            "Op:Add({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Sub(a, b) => format!(
            "Op:Sub({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Mul(a, b) => format!(
            "Op:Mul({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Div(a, b) => format!(
            "Op:Div({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Pow(a, b) => format!(
            "Op:Pow({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Mod(a, b) => format!(
            "Op:Mod({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Min(a, b) => format!(
            "Op:Min({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Max(a, b) => format!(
            "Op:Max({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Eq(a, b) => format!(
            "Op:Eq({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Lt(a, b) => format!(
            "Op:Lt({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Gt(a, b) => format!(
            "Op:Gt({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Lte(a, b) => format!(
            "Op:Lte({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        TLExpr::Gte(a, b) => format!(
            "Op:Gte({},{})",
            canonical_order_key(a),
            canonical_order_key(b)
        ),
        // Unary math
        TLExpr::Abs(inner) => format!("Op:Abs({})", canonical_order_key(inner)),
        TLExpr::Floor(inner) => format!("Op:Floor({})", canonical_order_key(inner)),
        TLExpr::Ceil(inner) => format!("Op:Ceil({})", canonical_order_key(inner)),
        TLExpr::Round(inner) => format!("Op:Round({})", canonical_order_key(inner)),
        TLExpr::Sqrt(inner) => format!("Op:Sqrt({})", canonical_order_key(inner)),
        TLExpr::Exp(inner) => format!("Op:Exp({})", canonical_order_key(inner)),
        TLExpr::Log(inner) => format!("Op:Log({})", canonical_order_key(inner)),
        TLExpr::Sin(inner) => format!("Op:Sin({})", canonical_order_key(inner)),
        TLExpr::Cos(inner) => format!("Op:Cos({})", canonical_order_key(inner)),
        TLExpr::Tan(inner) => format!("Op:Tan({})", canonical_order_key(inner)),
        TLExpr::EmptySet => "L:EmptySet".to_string(),
        TLExpr::Nominal { name } => format!("L:Nominal({})", name),
        // For all other complex variants, use Debug for deterministic ordering
        other => format!("X:{:?}", other),
    }
}

/// Convenience function: canonicalize and return the result.
pub fn canonicalize(expr: &TLExpr) -> (TLExpr, CanonicalStats) {
    Canonicalizer::new().canonicalize(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    fn pred_a() -> TLExpr {
        TLExpr::pred("a", vec![Term::var("x")])
    }

    fn pred_b() -> TLExpr {
        TLExpr::pred("b", vec![Term::var("x")])
    }

    fn pred_c() -> TLExpr {
        TLExpr::pred("c", vec![Term::var("x")])
    }

    #[test]
    fn test_double_neg_elimination() {
        let p = pred_a();
        let expr = TLExpr::negate(TLExpr::negate(p.clone()));
        let (result, stats) = canonicalize(&expr);
        assert_eq!(result, p);
        assert_eq!(stats.double_neg_removed, 1);
    }

    #[test]
    fn test_double_neg_nested_three() {
        // NOT(NOT(NOT(pred))) → NOT(pred)
        let p = pred_a();
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::negate(p.clone())));
        let (result, _stats) = canonicalize(&expr);
        assert_eq!(result, TLExpr::negate(p));
    }

    #[test]
    fn test_and_commutative_sorted() {
        let a = pred_a();
        let b = pred_b();
        let c = Canonicalizer::new();
        let key1 = c.canonical_key(&TLExpr::and(b.clone(), a.clone()));
        let key2 = c.canonical_key(&TLExpr::and(a.clone(), b.clone()));
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_or_commutative_sorted() {
        let a = pred_a();
        let b = pred_b();
        let c = Canonicalizer::new();
        let key1 = c.canonical_key(&TLExpr::or(b.clone(), a.clone()));
        let key2 = c.canonical_key(&TLExpr::or(a.clone(), b.clone()));
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_nested_and_consistent() {
        // AND(AND(a,b), c) key == AND(a, AND(b,c)) key (both flatten + sort)
        let a = pred_a();
        let b = pred_b();
        let c = pred_c();
        let can = Canonicalizer::new();
        let left_nested = TLExpr::and(TLExpr::and(a.clone(), b.clone()), c.clone());
        let right_nested = TLExpr::and(a.clone(), TLExpr::and(b.clone(), c.clone()));
        let key1 = can.canonical_key(&left_nested);
        let key2 = can.canonical_key(&right_nested);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_canonical_key_deterministic() {
        let expr = TLExpr::and(pred_a(), TLExpr::or(pred_b(), pred_c()));
        let c = Canonicalizer::new();
        let key1 = c.canonical_key(&expr);
        let key2 = c.canonical_key(&expr);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_canonical_key_different_exprs() {
        let c = Canonicalizer::new();
        let key1 = c.canonical_key(&TLExpr::and(pred_a(), pred_b()));
        let key2 = c.canonical_key(&TLExpr::or(pred_a(), pred_b()));
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_stats_double_neg_counted() {
        let expr = TLExpr::negate(TLExpr::negate(pred_a()));
        let (_result, stats) = canonicalize(&expr);
        assert_eq!(stats.double_neg_removed, 1);
        assert!(stats.total_rewrites >= 1);
    }

    #[test]
    fn test_stats_commutative_counted() {
        // AND(b, a) should sort to AND(a, b), incrementing commutative_sorted
        let a = pred_a();
        let b = pred_b();
        let expr = TLExpr::and(b, a);
        let (_result, stats) = canonicalize(&expr);
        assert_eq!(stats.commutative_sorted, 1);
    }

    #[test]
    fn test_stats_merge() {
        let mut s1 = CanonicalStats {
            double_neg_removed: 2,
            commutative_sorted: 1,
            nested_flattened: 3,
            total_rewrites: 6,
        };
        let s2 = CanonicalStats {
            double_neg_removed: 1,
            commutative_sorted: 4,
            nested_flattened: 0,
            total_rewrites: 5,
        };
        s1.merge(&s2);
        assert_eq!(s1.double_neg_removed, 3);
        assert_eq!(s1.commutative_sorted, 5);
        assert_eq!(s1.nested_flattened, 3);
        assert_eq!(s1.total_rewrites, 11);
    }

    #[test]
    fn test_canonicalize_pred_unchanged() {
        let p = pred_a();
        let (result, stats) = canonicalize(&p);
        assert_eq!(result, p);
        assert_eq!(stats.total_rewrites, 0);
    }

    #[test]
    fn test_canonicalize_constant_unchanged() {
        let c = TLExpr::Constant(42.0);
        let (result, stats) = canonicalize(&c);
        assert_eq!(result, c);
        assert_eq!(stats.total_rewrites, 0);
    }

    #[test]
    fn test_canonicalize_exists_recurses() {
        // exists body containing double neg should be canonicalized
        let body = TLExpr::negate(TLExpr::negate(pred_a()));
        let expr = TLExpr::exists("x", "D", body);
        let (result, stats) = canonicalize(&expr);
        assert_eq!(stats.double_neg_removed, 1);
        if let TLExpr::Exists { body, .. } = &result {
            assert!(matches!(body.as_ref(), TLExpr::Pred { .. }));
        } else {
            panic!("Expected Exists");
        }
    }

    #[test]
    fn test_canonicalize_forall_recurses() {
        let body = TLExpr::negate(TLExpr::negate(pred_a()));
        let expr = TLExpr::forall("x", "D", body);
        let (result, stats) = canonicalize(&expr);
        assert_eq!(stats.double_neg_removed, 1);
        if let TLExpr::ForAll { body, .. } = &result {
            assert!(matches!(body.as_ref(), TLExpr::Pred { .. }));
        } else {
            panic!("Expected ForAll");
        }
    }

    #[test]
    fn test_canonicalize_implication_recurses() {
        let premise = TLExpr::negate(TLExpr::negate(pred_a()));
        let conclusion = TLExpr::negate(TLExpr::negate(pred_b()));
        let expr = TLExpr::imply(premise, conclusion);
        let (result, stats) = canonicalize(&expr);
        assert_eq!(stats.double_neg_removed, 2);
        if let TLExpr::Imply(a, b) = &result {
            assert!(matches!(a.as_ref(), TLExpr::Pred { .. }));
            assert!(matches!(b.as_ref(), TLExpr::Pred { .. }));
        } else {
            panic!("Expected Imply");
        }
    }

    #[test]
    fn test_canonicalize_deep_nesting() {
        // Build a deeply nested expression: AND(AND(AND(...), b), c)
        let mut expr = pred_a();
        for i in 0..50 {
            let p = TLExpr::pred(format!("p{}", i), vec![Term::var("x")]);
            expr = TLExpr::and(expr, p);
        }
        // Should not stack overflow
        let (result, _stats) = canonicalize(&expr);
        // Result should be valid (just check it doesn't panic)
        let _ = canonical_order_key(&result);
    }

    #[test]
    fn test_canonical_order_key_pred() {
        let p = pred_a();
        let key = canonical_order_key(&p);
        assert!(
            key.starts_with("P:"),
            "Expected key to start with 'P:', got: {}",
            key
        );
        assert!(key.contains("a"));
    }

    #[test]
    fn test_canonical_order_key_constant() {
        let c = TLExpr::Constant(42.5);
        let key = canonical_order_key(&c);
        assert!(
            key.starts_with("C:"),
            "Expected key to start with 'C:', got: {}",
            key
        );
    }

    #[test]
    fn test_convenience_fn() {
        let expr = TLExpr::negate(TLExpr::negate(pred_a()));
        let (result, stats) = canonicalize(&expr);
        assert_eq!(result, pred_a());
        assert_eq!(stats.double_neg_removed, 1);
    }

    #[test]
    fn test_disabled_rules() {
        let a = pred_a();
        let b = pred_b();
        // With sort disabled, AND(b, a) should NOT be sorted
        let c = Canonicalizer::new().with_sort_commutative(false);
        let expr = TLExpr::and(b.clone(), a.clone());
        let (result, stats) = c.canonicalize(&expr);
        assert_eq!(stats.commutative_sorted, 0);
        // The result should still be AND(b, a), not AND(a, b)
        if let TLExpr::And(left, right) = &result {
            assert_eq!(left.as_ref(), &b);
            assert_eq!(right.as_ref(), &a);
        } else {
            panic!("Expected And");
        }
    }
}
