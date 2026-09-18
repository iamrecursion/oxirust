use tensorlogic_ir::TLExpr;

/// Substitute all free occurrences of `var` with `replacement` in `body`.
///
/// This is capture-avoiding: when a binder that re-introduces `var` is
/// encountered, substitution stops for the sub-tree guarded by that binder.
///
/// Variable occurrences are zero-argument predicates with `name == var`
/// and `Term::Var(var)` inside predicate arguments.
pub fn substitute(var: &str, replacement: &TLExpr, body: TLExpr) -> TLExpr {
    subst(var, replacement, body)
}

pub(crate) fn subst(var: &str, repl: &TLExpr, expr: TLExpr) -> TLExpr {
    match expr {
        // Zero-arg predicate used as a variable reference.
        TLExpr::Pred { ref name, ref args } if args.is_empty() && name == var => repl.clone(),

        // Predicate with arguments: substitute in Term::Var occurrences.
        TLExpr::Pred { name, args } => {
            let new_args = args
                .into_iter()
                .map(|t| match &t {
                    tensorlogic_ir::Term::Var(v) if v == var => {
                        // We can only substitute if replacement is a zero-arg
                        // Pred (variable) or Constant; otherwise keep the Term.
                        match repl {
                            TLExpr::Pred { name: rn, args: ra } if ra.is_empty() => {
                                tensorlogic_ir::Term::Var(rn.clone())
                            }
                            _ => t,
                        }
                    }
                    _ => t,
                })
                .collect();
            TLExpr::Pred {
                name,
                args: new_args,
            }
        }

        // ── Binary nodes ─────────────────────────────────────────────────
        TLExpr::And(l, r) => TLExpr::And(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Or(l, r) => TLExpr::Or(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Imply(l, r) => TLExpr::Imply(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Add(l, r) => TLExpr::Add(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Sub(l, r) => TLExpr::Sub(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Mul(l, r) => TLExpr::Mul(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Div(l, r) => TLExpr::Div(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Pow(l, r) => TLExpr::Pow(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Mod(l, r) => TLExpr::Mod(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Min(l, r) => TLExpr::Min(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Max(l, r) => TLExpr::Max(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Eq(l, r) => TLExpr::Eq(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Lt(l, r) => TLExpr::Lt(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Gt(l, r) => TLExpr::Gt(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Lte(l, r) => TLExpr::Lte(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),
        TLExpr::Gte(l, r) => TLExpr::Gte(
            Box::new(subst(var, repl, *l)),
            Box::new(subst(var, repl, *r)),
        ),

        // ── Unary nodes ──────────────────────────────────────────────────
        TLExpr::Not(e) => TLExpr::Not(Box::new(subst(var, repl, *e))),
        TLExpr::Score(e) => TLExpr::Score(Box::new(subst(var, repl, *e))),
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(subst(var, repl, *e))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(subst(var, repl, *e))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(subst(var, repl, *e))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(subst(var, repl, *e))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(subst(var, repl, *e))),
        TLExpr::Exp(e) => TLExpr::Exp(Box::new(subst(var, repl, *e))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(subst(var, repl, *e))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(subst(var, repl, *e))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(subst(var, repl, *e))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(subst(var, repl, *e))),
        TLExpr::Box(e) => TLExpr::Box(Box::new(subst(var, repl, *e))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(subst(var, repl, *e))),
        TLExpr::Next(e) => TLExpr::Next(Box::new(subst(var, repl, *e))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(subst(var, repl, *e))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(subst(var, repl, *e))),

        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind,
            expr: Box::new(subst(var, repl, *expr)),
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight,
            rule: Box::new(subst(var, repl, *rule)),
        },

        // ── Temporal / logical binary ─────────────────────────────────────
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(subst(var, repl, *before)),
            after: Box::new(subst(var, repl, *after)),
        },
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(subst(var, repl, *released)),
            releaser: Box::new(subst(var, repl, *releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(subst(var, repl, *before)),
            after: Box::new(subst(var, repl, *after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(subst(var, repl, *released)),
            releaser: Box::new(subst(var, repl, *releaser)),
        },

        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind,
            left: Box::new(subst(var, repl, *left)),
            right: Box::new(subst(var, repl, *right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind,
            left: Box::new(subst(var, repl, *left)),
            right: Box::new(subst(var, repl, *right)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind,
            premise: Box::new(subst(var, repl, *premise)),
            conclusion: Box::new(subst(var, repl, *conclusion)),
        },

        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .into_iter()
                .map(|(p, e)| (p, subst(var, repl, e)))
                .collect(),
        },

        // ── IfThenElse ────────────────────────────────────────────────────
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(subst(var, repl, *condition)),
            then_branch: Box::new(subst(var, repl, *then_branch)),
            else_branch: Box::new(subst(var, repl, *else_branch)),
        },

        // ── Binders — capture-avoiding ────────────────────────────────────
        TLExpr::Exists {
            var: binder,
            domain,
            body,
        } => {
            if binder == var {
                TLExpr::Exists {
                    var: binder,
                    domain,
                    body,
                }
            } else {
                TLExpr::Exists {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::ForAll {
            var: binder,
            domain,
            body,
        } => {
            if binder == var {
                TLExpr::ForAll {
                    var: binder,
                    domain,
                    body,
                }
            } else {
                TLExpr::ForAll {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::SoftExists {
            var: binder,
            domain,
            body,
            temperature,
        } => {
            if binder == var {
                TLExpr::SoftExists {
                    var: binder,
                    domain,
                    body,
                    temperature,
                }
            } else {
                TLExpr::SoftExists {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                    temperature,
                }
            }
        }
        TLExpr::SoftForAll {
            var: binder,
            domain,
            body,
            temperature,
        } => {
            if binder == var {
                TLExpr::SoftForAll {
                    var: binder,
                    domain,
                    body,
                    temperature,
                }
            } else {
                TLExpr::SoftForAll {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                    temperature,
                }
            }
        }
        TLExpr::Aggregate {
            op,
            var: binder,
            domain,
            body,
            group_by,
        } => {
            if binder == var {
                TLExpr::Aggregate {
                    op,
                    var: binder,
                    domain,
                    body,
                    group_by,
                }
            } else {
                TLExpr::Aggregate {
                    op,
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                    group_by,
                }
            }
        }
        // Let: substitute in value unconditionally (outer scope), substitute
        // in body only if binder != var (capture avoidance).
        TLExpr::Let {
            var: binder,
            value,
            body,
        } => {
            let new_value = subst(var, repl, *value);
            if binder == var {
                TLExpr::Let {
                    var: binder,
                    value: Box::new(new_value),
                    body,
                }
            } else {
                TLExpr::Let {
                    var: binder,
                    value: Box::new(new_value),
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::Lambda {
            var: binder,
            var_type,
            body,
        } => {
            if binder == var {
                TLExpr::Lambda {
                    var: binder,
                    var_type,
                    body,
                }
            } else {
                TLExpr::Lambda {
                    var: binder,
                    var_type,
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::CountingExists {
            var: binder,
            domain,
            body,
            min_count,
        } => {
            if binder == var {
                TLExpr::CountingExists {
                    var: binder,
                    domain,
                    body,
                    min_count,
                }
            } else {
                TLExpr::CountingExists {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                    min_count,
                }
            }
        }
        TLExpr::CountingForAll {
            var: binder,
            domain,
            body,
            min_count,
        } => {
            if binder == var {
                TLExpr::CountingForAll {
                    var: binder,
                    domain,
                    body,
                    min_count,
                }
            } else {
                TLExpr::CountingForAll {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                    min_count,
                }
            }
        }
        TLExpr::ExactCount {
            var: binder,
            domain,
            body,
            count,
        } => {
            if binder == var {
                TLExpr::ExactCount {
                    var: binder,
                    domain,
                    body,
                    count,
                }
            } else {
                TLExpr::ExactCount {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                    count,
                }
            }
        }
        TLExpr::Majority {
            var: binder,
            domain,
            body,
        } => {
            if binder == var {
                TLExpr::Majority {
                    var: binder,
                    domain,
                    body,
                }
            } else {
                TLExpr::Majority {
                    var: binder,
                    domain,
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::LeastFixpoint { var: binder, body } => {
            if binder == var {
                TLExpr::LeastFixpoint { var: binder, body }
            } else {
                TLExpr::LeastFixpoint {
                    var: binder,
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::GreatestFixpoint { var: binder, body } => {
            if binder == var {
                TLExpr::GreatestFixpoint { var: binder, body }
            } else {
                TLExpr::GreatestFixpoint {
                    var: binder,
                    body: Box::new(subst(var, repl, *body)),
                }
            }
        }
        TLExpr::SetComprehension {
            var: binder,
            domain,
            condition,
        } => {
            if binder == var {
                TLExpr::SetComprehension {
                    var: binder,
                    domain,
                    condition,
                }
            } else {
                TLExpr::SetComprehension {
                    var: binder,
                    domain,
                    condition: Box::new(subst(var, repl, *condition)),
                }
            }
        }

        // ── Set operations ────────────────────────────────────────────────
        TLExpr::Apply { function, argument } => TLExpr::Apply {
            function: Box::new(subst(var, repl, *function)),
            argument: Box::new(subst(var, repl, *argument)),
        },
        TLExpr::SetMembership { element, set } => TLExpr::SetMembership {
            element: Box::new(subst(var, repl, *element)),
            set: Box::new(subst(var, repl, *set)),
        },
        TLExpr::SetUnion { left, right } => TLExpr::SetUnion {
            left: Box::new(subst(var, repl, *left)),
            right: Box::new(subst(var, repl, *right)),
        },
        TLExpr::SetIntersection { left, right } => TLExpr::SetIntersection {
            left: Box::new(subst(var, repl, *left)),
            right: Box::new(subst(var, repl, *right)),
        },
        TLExpr::SetDifference { left, right } => TLExpr::SetDifference {
            left: Box::new(subst(var, repl, *left)),
            right: Box::new(subst(var, repl, *right)),
        },
        TLExpr::SetCardinality { set } => TLExpr::SetCardinality {
            set: Box::new(subst(var, repl, *set)),
        },

        TLExpr::At { nominal, formula } => TLExpr::At {
            nominal,
            formula: Box::new(subst(var, repl, *formula)),
        },
        TLExpr::Somewhere { formula } => TLExpr::Somewhere {
            formula: Box::new(subst(var, repl, *formula)),
        },
        TLExpr::Everywhere { formula } => TLExpr::Everywhere {
            formula: Box::new(subst(var, repl, *formula)),
        },
        TLExpr::Explain { formula } => TLExpr::Explain {
            formula: Box::new(subst(var, repl, *formula)),
        },

        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::GlobalCardinality {
            variables,
            values: values.into_iter().map(|e| subst(var, repl, e)).collect(),
            min_occurrences,
            max_occurrences,
        },

        // ── Leaves ───────────────────────────────────────────────────────
        leaf @ (TLExpr::Constant(_)
        | TLExpr::EmptySet
        | TLExpr::AllDifferent { .. }
        | TLExpr::Nominal { .. }
        | TLExpr::Abducible { .. }
        | TLExpr::SymbolLiteral(_)) => leaf,

        TLExpr::Match { scrutinee, arms } => TLExpr::Match {
            scrutinee: Box::new(subst(var, repl, *scrutinee)),
            arms: arms
                .into_iter()
                .map(|(pat, body)| (pat, Box::new(subst(var, repl, *body))))
                .collect(),
        },
    }
}
