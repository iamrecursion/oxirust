//! # Literals for Resolution
//!
//! A literal is an atomic formula or its negation. This module defines the
//! [`Literal`] type, its unification and matching operations, and supporting
//! helpers used by subsumption and resolution.

use crate::expr::TLExpr;
use crate::term::Term;
use crate::unification::{unify_term_list, Substitution};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A literal is an atomic formula or its negation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Literal {
    /// The underlying atomic formula
    pub atom: TLExpr,
    /// True if positive literal, false if negative
    pub polarity: bool,
}

impl Literal {
    /// Create a positive literal from an atomic formula.
    pub fn positive(atom: TLExpr) -> Self {
        Literal {
            atom,
            polarity: true,
        }
    }

    /// Create a negative literal from an atomic formula.
    pub fn negative(atom: TLExpr) -> Self {
        Literal {
            atom,
            polarity: false,
        }
    }

    /// Negate this literal.
    pub fn negate(&self) -> Self {
        Literal {
            atom: self.atom.clone(),
            polarity: !self.polarity,
        }
    }

    /// Check if this literal is complementary to another (same atom, opposite polarity).
    ///
    /// For ground literals (no variables), this checks exact equality.
    pub fn is_complementary(&self, other: &Literal) -> bool {
        self.atom == other.atom && self.polarity != other.polarity
    }

    /// Attempt to unify this literal with another for resolution.
    ///
    /// Returns the most general unifier (MGU) if the atoms can be unified
    /// and the polarities are opposite.
    ///
    /// This is used for first-order resolution with variables.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal};
    ///
    /// // P(x) and ¬P(a) can be unified with {x/a}
    /// let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    /// let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
    ///
    /// let mgu = p_x.try_unify(&not_p_a);
    /// assert!(mgu.is_some());
    /// ```
    pub fn try_unify(&self, other: &Literal) -> Option<Substitution> {
        // Must have opposite polarity
        if self.polarity == other.polarity {
            return None;
        }

        // Try to unify the atoms
        self.try_unify_atoms(&other.atom)
    }

    /// Attempt to unify this literal's atom with another expression.
    ///
    /// This extracts terms from predicates and attempts unification.
    fn try_unify_atoms(&self, other_atom: &TLExpr) -> Option<Substitution> {
        match (&self.atom, other_atom) {
            (
                TLExpr::Pred {
                    name: n1,
                    args: args1,
                },
                TLExpr::Pred {
                    name: n2,
                    args: args2,
                },
            ) => {
                // Predicate names must match
                if n1 != n2 {
                    return None;
                }

                // Arity must match
                if args1.len() != args2.len() {
                    return None;
                }

                // Unify argument lists
                let pairs: Vec<(Term, Term)> = args1
                    .iter()
                    .zip(args2.iter())
                    .map(|(t1, t2)| (t1.clone(), t2.clone()))
                    .collect();

                unify_term_list(&pairs).ok()
            }
            _ => None,
        }
    }

    /// Apply a substitution to this literal.
    ///
    /// This creates a new literal with the substitution applied to all terms.
    pub fn apply_substitution(&self, subst: &Substitution) -> Literal {
        let new_atom = self.apply_subst_to_expr(&self.atom, subst);
        Literal {
            atom: new_atom,
            polarity: self.polarity,
        }
    }

    /// Apply substitution to an expression (helper for apply_substitution).
    fn apply_subst_to_expr(&self, expr: &TLExpr, subst: &Substitution) -> TLExpr {
        match expr {
            TLExpr::Pred { name, args } => {
                let new_args = args.iter().map(|term| subst.apply(term)).collect();
                TLExpr::Pred {
                    name: name.clone(),
                    args: new_args,
                }
            }
            // For other expression types, return as-is
            _ => expr.clone(),
        }
    }

    /// Check if this is a positive literal.
    pub fn is_positive(&self) -> bool {
        self.polarity
    }

    /// Check if this is a negative literal.
    pub fn is_negative(&self) -> bool {
        !self.polarity
    }

    /// Get the free variables in this literal.
    pub fn free_vars(&self) -> HashSet<String> {
        self.atom.free_vars()
    }

    /// Try one-way matching for subsumption: only variables in `allowed_vars` can be bound.
    ///
    /// This is different from full unification - we only allow variables from the
    /// subsuming clause to be instantiated, not variables from the subsumed clause.
    pub(super) fn try_one_way_match(
        &self,
        other_atom: &TLExpr,
        allowed_vars: &HashSet<String>,
    ) -> Option<Substitution> {
        match (&self.atom, other_atom) {
            (
                TLExpr::Pred {
                    name: n1,
                    args: args1,
                },
                TLExpr::Pred {
                    name: n2,
                    args: args2,
                },
            ) => {
                // Predicate names must match
                if n1 != n2 {
                    return None;
                }

                // Arity must match
                if args1.len() != args2.len() {
                    return None;
                }

                // Try one-way matching for each argument pair
                let mut subst = Substitution::empty();

                for (t1, t2) in args1.iter().zip(args2.iter()) {
                    if !try_one_way_match_terms(t1, t2, allowed_vars, &mut subst) {
                        return None;
                    }
                }

                Some(subst)
            }
            _ => None,
        }
    }
}

/// One-way matching for terms: only variables in `allowed_vars` can be bound.
pub(super) fn try_one_way_match_terms(
    t1: &Term,
    t2: &Term,
    allowed_vars: &HashSet<String>,
    subst: &mut Substitution,
) -> bool {
    // Apply current substitution to t1
    let t1_subst = subst.apply(t1);

    match (&t1_subst, t2) {
        // Same constant
        (Term::Const(c1), Term::Const(c2)) => c1 == c2,

        // Same variable
        (Term::Var(v1), Term::Var(v2)) => v1 == v2,

        // t1 is an allowed variable, bind it to t2
        (Term::Var(v1), _) if allowed_vars.contains(v1) => {
            // Check if already bound by trying to apply the substitution
            let after_subst = subst.apply(&t1_subst);
            if after_subst != t1_subst {
                // Already bound, check if it matches t2
                &after_subst == t2
            } else {
                // Not bound, bind it now
                subst.bind(v1.clone(), t2.clone());
                true
            }
        }

        // t1 is a variable but not allowed to be bound
        (Term::Var(_), _) => false,

        // t2 is a variable but we can't bind it (one-way matching)
        (_, Term::Var(_)) => false,

        // Both typed with same type
        (
            Term::Typed {
                value: inner1,
                type_annotation: ty1,
            },
            Term::Typed {
                value: inner2,
                type_annotation: ty2,
            },
        ) => {
            if ty1 != ty2 {
                return false;
            }
            try_one_way_match_terms(inner1, inner2, allowed_vars, subst)
        }
        (Term::Typed { value, .. }, other) | (other, Term::Typed { value, .. }) => {
            try_one_way_match_terms(value, other, allowed_vars, subst)
        }
    }
}
