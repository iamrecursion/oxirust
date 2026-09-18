//! # Clauses for Resolution
//!
//! A [`Clause`] is a disjunction of literals. This module implements
//! construction, tautology/Horn detection, substitution, renaming, and
//! theta-subsumption used by the resolution prover.

use crate::expr::TLExpr;
use crate::term::Term;
use crate::unification::{rename_vars, Substitution};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::literal::Literal;

/// A clause is a disjunction of literals: `L₁ ∨ L₂ ∨ ... ∨ Lₙ`.
///
/// Special cases:
/// - Empty clause (∅): contradiction, no literals
/// - Unit clause: single literal
/// - Horn clause: at most one positive literal
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clause {
    /// The literals in this clause (disjunction)
    pub literals: Vec<Literal>,
}

impl Clause {
    /// Create a new clause from a list of literals.
    pub fn from_literals(literals: Vec<Literal>) -> Self {
        // Remove duplicates and sort for consistency
        let mut unique_lits: Vec<Literal> = literals.into_iter().collect();
        unique_lits.sort_by(|a, b| {
            let a_str = format!("{:?}", a);
            let b_str = format!("{:?}", b);
            a_str.cmp(&b_str)
        });
        unique_lits.dedup();

        Clause {
            literals: unique_lits,
        }
    }

    /// Create an empty clause (contradiction).
    pub fn empty() -> Self {
        Clause { literals: vec![] }
    }

    /// Create a unit clause (single literal).
    pub fn unit(literal: Literal) -> Self {
        Clause {
            literals: vec![literal],
        }
    }

    /// Check if this is the empty clause (contradiction).
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if this is a unit clause (single literal).
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Check if this is a Horn clause (at most one positive literal).
    pub fn is_horn(&self) -> bool {
        self.literals.iter().filter(|l| l.is_positive()).count() <= 1
    }

    /// Get the number of literals in this clause.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if clause is empty (different from is_empty which checks for contradiction).
    pub fn is_len_zero(&self) -> bool {
        self.literals.is_empty()
    }

    /// Get all free variables in this clause.
    pub fn free_vars(&self) -> HashSet<String> {
        self.literals
            .iter()
            .flat_map(|lit| lit.free_vars())
            .collect()
    }

    /// Check if this clause subsumes another (is more general).
    ///
    /// **Theta-Subsumption**: Clause C subsumes D (C ⪯ D) if there exists a
    /// substitution θ such that Cθ ⊆ D.
    ///
    /// This means C is more general than D. For example:
    /// - `{P(x)}` subsumes `{P(a), Q(a)}` with θ = {x/a}
    /// - `{P(x), Q(x)}` subsumes `{P(a), Q(a), R(a)}` with θ = {x/a}
    ///
    /// # Implementation
    ///
    /// We try to find a substitution by:
    /// 1. For each literal in C, try to unify it with some literal in D
    /// 2. Check if all substitutions are consistent
    /// 3. If successful, C subsumes D
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal, Clause};
    ///
    /// // {P(x)} subsumes {P(a)}
    /// let c = Clause::unit(Literal::positive(TLExpr::pred("P", vec![Term::var("x")])));
    /// let d = Clause::unit(Literal::positive(TLExpr::pred("P", vec![Term::constant("a")])));
    /// assert!(c.subsumes(&d));
    ///
    /// // {P(a)} does not subsume {P(x)} (x is more general than a)
    /// assert!(!d.subsumes(&c));
    /// ```
    pub fn subsumes(&self, other: &Clause) -> bool {
        // Empty clause subsumes nothing (except itself)
        if self.is_empty() {
            return other.is_empty();
        }

        // Can't subsume if C has more literals than D
        if self.literals.len() > other.literals.len() {
            return false;
        }

        // Try to find a substitution that makes all of C's literals match some literal in D
        self.try_subsumption_matching(other).is_some()
    }

    /// Attempt to find a substitution θ such that Cθ ⊆ D.
    ///
    /// Uses a backtracking search to find consistent literal matchings.
    fn try_subsumption_matching(&self, other: &Clause) -> Option<Substitution> {
        // Rename variables in self to avoid conflicts
        static mut SUBSUME_COUNTER: usize = 0;
        let counter = unsafe {
            SUBSUME_COUNTER += 1;
            SUBSUME_COUNTER
        };

        let renamed_self = self.rename_variables(&format!("_s{}", counter));
        let renamed_vars: HashSet<String> = renamed_self.free_vars();

        // Try to match each literal in renamed_self with literals in other
        let mut subst = Substitution::empty();

        for self_lit in &renamed_self.literals {
            // Try to find a matching literal in other
            let mut found_match = false;

            for other_lit in &other.literals {
                // Literals must have the same polarity
                if self_lit.polarity != other_lit.polarity {
                    continue;
                }

                // Try one-way matching: only variables from self can be bound
                if let Some(lit_mgu) = self_lit.try_one_way_match(&other_lit.atom, &renamed_vars) {
                    // Check if this unifier is consistent with existing substitution
                    if let Ok(()) = subst.try_extend(&lit_mgu) {
                        found_match = true;
                        break;
                    }
                }
            }

            if !found_match {
                return None; // Failed to match this literal
            }
        }

        Some(subst)
    }

    /// Check if this clause is tautology (contains complementary literals).
    pub fn is_tautology(&self) -> bool {
        for i in 0..self.literals.len() {
            for j in (i + 1)..self.literals.len() {
                if self.literals[i].is_complementary(&self.literals[j]) {
                    return true;
                }
            }
        }
        false
    }

    /// Apply a substitution to this clause.
    ///
    /// This creates a new clause with the substitution applied to all literals.
    pub fn apply_substitution(&self, subst: &Substitution) -> Clause {
        let new_literals = self
            .literals
            .iter()
            .map(|lit| lit.apply_substitution(subst))
            .collect();
        Clause::from_literals(new_literals)
    }

    /// Rename all variables in this clause with a suffix.
    ///
    /// This is used for standardizing apart clauses before resolution
    /// to avoid variable name conflicts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal, Clause};
    ///
    /// // P(x) ∨ Q(x)
    /// let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    /// let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    /// let clause = Clause::from_literals(vec![p_x, q_x]);
    ///
    /// // Rename to P(x_1) ∨ Q(x_1)
    /// let renamed = clause.rename_variables("1");
    /// ```
    pub fn rename_variables(&self, suffix: &str) -> Clause {
        let renamed_literals = self
            .literals
            .iter()
            .map(|lit| self.rename_literal(lit, suffix))
            .collect();
        Clause::from_literals(renamed_literals)
    }

    /// Rename variables in a literal (helper for rename_variables).
    fn rename_literal(&self, lit: &Literal, suffix: &str) -> Literal {
        match &lit.atom {
            TLExpr::Pred { name, args } => {
                let renamed_args: Vec<Term> =
                    args.iter().map(|term| rename_vars(term, suffix)).collect();
                Literal {
                    atom: TLExpr::Pred {
                        name: name.clone(),
                        args: renamed_args,
                    },
                    polarity: lit.polarity,
                }
            }
            _ => lit.clone(),
        }
    }
}
