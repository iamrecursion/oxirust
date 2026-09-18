//! # Resolution Prover
//!
//! The [`ResolutionProver`] drives refutation-based theorem proving over a
//! set of clauses. It supports multiple strategies: saturation, set-of-support,
//! unit resolution, and linear resolution. First-order binary resolution with
//! unification and standardizing-apart is also provided.

use std::collections::VecDeque;

use super::clause::Clause;
use super::literal::Literal;
use super::proof::{ProofResult, ProverStats, ResolutionStep, ResolutionStrategy};

/// Resolution-based theorem prover.
pub struct ResolutionProver {
    /// Initial clause set
    pub(super) clauses: Vec<Clause>,
    /// Strategy to use
    strategy: ResolutionStrategy,
    /// Statistics
    pub stats: ProverStats,
}

impl ResolutionProver {
    /// Create a new resolution prover with default strategy.
    pub fn new() -> Self {
        ResolutionProver {
            clauses: Vec::new(),
            strategy: ResolutionStrategy::Saturation { max_clauses: 10000 },
            stats: ProverStats::default(),
        }
    }

    /// Create a prover with a specific strategy.
    pub fn with_strategy(strategy: ResolutionStrategy) -> Self {
        ResolutionProver {
            clauses: Vec::new(),
            strategy,
            stats: ProverStats::default(),
        }
    }

    /// Add a clause to the initial clause set.
    pub fn add_clause(&mut self, clause: Clause) {
        // Don't add tautologies
        if !clause.is_tautology() {
            self.clauses.push(clause);
        } else {
            self.stats.tautologies_removed += 1;
        }
    }

    /// Add multiple clauses.
    pub fn add_clauses(&mut self, clauses: Vec<Clause>) {
        for clause in clauses {
            self.add_clause(clause);
        }
    }

    /// Reset the prover (clear clauses and stats).
    pub fn reset(&mut self) {
        self.clauses.clear();
        self.stats = ProverStats::default();
    }

    /// Perform binary resolution on two clauses.
    ///
    /// Returns all possible resolvents.
    ///
    /// This is the ground resolution (no variables). For first-order resolution
    /// with variables, use `resolve_first_order`.
    fn resolve(&self, c1: &Clause, c2: &Clause) -> Vec<(Clause, Literal)> {
        let mut resolvents = Vec::new();

        // Try to resolve on each pair of complementary literals
        for lit1 in &c1.literals {
            for lit2 in &c2.literals {
                if lit1.is_complementary(lit2) {
                    // Build resolvent: (c1 - lit1) ∪ (c2 - lit2)
                    let mut new_literals = Vec::new();

                    // Add literals from c1 except lit1
                    for lit in &c1.literals {
                        if lit != lit1 {
                            new_literals.push(lit.clone());
                        }
                    }

                    // Add literals from c2 except lit2
                    for lit in &c2.literals {
                        if lit != lit2 {
                            new_literals.push(lit.clone());
                        }
                    }

                    let resolvent = Clause::from_literals(new_literals);
                    resolvents.push((resolvent, lit1.clone()));
                }
            }
        }

        resolvents
    }

    /// Perform first-order binary resolution with unification.
    ///
    /// This supports resolution on clauses with variables by using unification.
    /// Clauses are standardized apart before resolution to avoid variable conflicts.
    ///
    /// Returns all possible resolvents with their MGUs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_ir::{TLExpr, Term, Literal, Clause, ResolutionProver};
    ///
    /// // {P(x)} and {¬P(a)} resolve to {} (empty clause) with MGU {x/a}
    /// let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    /// let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
    ///
    /// let c1 = Clause::unit(p_x);
    /// let c2 = Clause::unit(not_p_a);
    ///
    /// let prover = ResolutionProver::new();
    /// let resolvents = prover.resolve_first_order(&c1, &c2);
    ///
    /// assert_eq!(resolvents.len(), 1);
    /// assert!(resolvents[0].0.is_empty()); // Empty clause derived
    /// ```
    pub fn resolve_first_order(&self, c1: &Clause, c2: &Clause) -> Vec<(Clause, Literal)> {
        // Use a simple counter for standardizing apart
        // In practice, this could use a global counter or timestamp
        static mut RENAME_COUNTER: usize = 0;
        let counter = unsafe {
            RENAME_COUNTER += 1;
            RENAME_COUNTER
        };

        // Standardize apart: rename variables to avoid conflicts
        let c1_renamed = c1.rename_variables(&format!("_c1_{}", counter));
        let c2_renamed = c2.rename_variables(&format!("_c2_{}", counter));

        let mut resolvents = Vec::new();

        // Try to unify each pair of opposite polarity literals
        for lit1 in &c1_renamed.literals {
            for lit2 in &c2_renamed.literals {
                // Try to unify with first-order unification
                if let Some(mgu) = lit1.try_unify(lit2) {
                    // Build resolvent: apply MGU to (c1 - lit1) ∪ (c2 - lit2)
                    let mut new_literals = Vec::new();

                    // Add literals from c1 except lit1, with MGU applied
                    for lit in &c1_renamed.literals {
                        if lit != lit1 {
                            new_literals.push(lit.apply_substitution(&mgu));
                        }
                    }

                    // Add literals from c2 except lit2, with MGU applied
                    for lit in &c2_renamed.literals {
                        if lit != lit2 {
                            new_literals.push(lit.apply_substitution(&mgu));
                        }
                    }

                    let resolvent = Clause::from_literals(new_literals);
                    // Return the original (non-renamed) literal for tracking
                    let orig_lit = lit1.clone(); // Could map back to original if needed
                    resolvents.push((resolvent, orig_lit));
                }
            }
        }

        resolvents
    }

    /// Check if a clause is subsumed by any clause in the set.
    fn is_subsumed(&self, clause: &Clause, clause_set: &[Clause]) -> bool {
        clause_set.iter().any(|c| c.subsumes(clause))
    }

    /// Attempt to prove the clause set unsatisfiable using resolution.
    pub fn prove(&mut self) -> ProofResult {
        match &self.strategy {
            ResolutionStrategy::Saturation { max_clauses } => self.prove_saturation(*max_clauses),
            ResolutionStrategy::SetOfSupport { max_steps } => self.prove_set_of_support(*max_steps),
            ResolutionStrategy::UnitResolution { max_steps } => {
                self.prove_unit_resolution(*max_steps)
            }
            ResolutionStrategy::Linear { max_depth } => self.prove_linear(*max_depth),
        }
    }

    /// Saturation-based proof: generate all resolvents.
    fn prove_saturation(&mut self, max_clauses: usize) -> ProofResult {
        let mut clause_set: Vec<Clause> = self.clauses.clone();
        let mut derivation = Vec::new();
        let mut steps = 0;

        // Check if empty clause is in initial set
        if clause_set.iter().any(|c| c.is_empty()) {
            self.stats.empty_clause_found = true;
            return ProofResult::Unsatisfiable {
                steps: 0,
                derivation: vec![],
            };
        }

        loop {
            let current_clauses: Vec<Clause> = clause_set.clone();
            let mut new_clauses = Vec::new();

            // Generate all resolvents
            for i in 0..current_clauses.len() {
                for j in (i + 1)..current_clauses.len() {
                    let resolvents = self.resolve(&current_clauses[i], &current_clauses[j]);

                    for (resolvent, resolved_lit) in resolvents {
                        steps += 1;
                        self.stats.resolution_steps += 1;

                        // Skip tautologies
                        if resolvent.is_tautology() {
                            self.stats.tautologies_removed += 1;
                            continue;
                        }

                        // Check for empty clause
                        if resolvent.is_empty() {
                            self.stats.empty_clause_found = true;
                            derivation.push(ResolutionStep {
                                parent1: current_clauses[i].clone(),
                                parent2: current_clauses[j].clone(),
                                resolvent: resolvent.clone(),
                                resolved_literal: resolved_lit,
                            });
                            return ProofResult::Unsatisfiable { steps, derivation };
                        }

                        // Skip if subsumed
                        if self.is_subsumed(&resolvent, &current_clauses) {
                            self.stats.clauses_subsumed += 1;
                            continue;
                        }

                        // Add new clause if not already present
                        if !clause_set.contains(&resolvent) && !new_clauses.contains(&resolvent) {
                            new_clauses.push(resolvent.clone());
                            derivation.push(ResolutionStep {
                                parent1: current_clauses[i].clone(),
                                parent2: current_clauses[j].clone(),
                                resolvent,
                                resolved_literal: resolved_lit,
                            });
                        }
                    }
                }
            }

            // Check for saturation or limit
            if new_clauses.is_empty() {
                return ProofResult::Saturated {
                    clauses_generated: clause_set.len(),
                };
            }

            // Add new clauses to set
            for clause in new_clauses {
                clause_set.push(clause);
                self.stats.clauses_generated += 1;

                if clause_set.len() >= max_clauses {
                    return ProofResult::ResourceLimitReached {
                        steps_attempted: steps,
                    };
                }
            }
        }
    }

    /// Set-of-support proof strategy.
    fn prove_set_of_support(&mut self, max_steps: usize) -> ProofResult {
        // Simplified: treat last clause as support set
        if self.clauses.is_empty() {
            return ProofResult::Satisfiable;
        }

        let support = self
            .clauses
            .pop()
            .expect("clauses must be non-empty before pop");
        let mut sos: VecDeque<Clause> = VecDeque::new();
        sos.push_back(support);

        let usable: Vec<Clause> = self.clauses.clone();
        let mut derivation = Vec::new();
        let mut steps = 0;

        while let Some(current) = sos.pop_front() {
            if steps >= max_steps {
                return ProofResult::ResourceLimitReached {
                    steps_attempted: steps,
                };
            }

            if current.is_empty() {
                self.stats.empty_clause_found = true;
                return ProofResult::Unsatisfiable { steps, derivation };
            }

            // Resolve with usable clauses
            for usable_clause in &usable {
                let resolvents = self.resolve(&current, usable_clause);

                for (resolvent, resolved_lit) in resolvents {
                    steps += 1;
                    self.stats.resolution_steps += 1;

                    if resolvent.is_tautology() {
                        self.stats.tautologies_removed += 1;
                        continue;
                    }

                    if resolvent.is_empty() {
                        self.stats.empty_clause_found = true;
                        derivation.push(ResolutionStep {
                            parent1: current.clone(),
                            parent2: usable_clause.clone(),
                            resolvent: resolvent.clone(),
                            resolved_literal: resolved_lit,
                        });
                        return ProofResult::Unsatisfiable { steps, derivation };
                    }

                    sos.push_back(resolvent.clone());
                    self.stats.clauses_generated += 1;
                    derivation.push(ResolutionStep {
                        parent1: current.clone(),
                        parent2: usable_clause.clone(),
                        resolvent,
                        resolved_literal: resolved_lit,
                    });
                }
            }
        }

        ProofResult::Satisfiable
    }

    /// Unit resolution strategy (only resolve with unit clauses).
    fn prove_unit_resolution(&mut self, max_steps: usize) -> ProofResult {
        let mut clauses = self.clauses.clone();
        let mut derivation = Vec::new();
        let mut steps = 0;

        loop {
            if steps >= max_steps {
                return ProofResult::ResourceLimitReached {
                    steps_attempted: steps,
                };
            }

            // Find unit clauses
            let unit_clauses: Vec<Clause> =
                clauses.iter().filter(|c| c.is_unit()).cloned().collect();

            if unit_clauses.is_empty() {
                return ProofResult::Satisfiable;
            }

            let mut new_clauses = Vec::new();
            let mut found_new = false;

            // Resolve each unit clause with all clauses
            for unit in &unit_clauses {
                for clause in &clauses {
                    if clause.is_unit() && clause == unit {
                        continue; // Skip self-resolution
                    }

                    let resolvents = self.resolve(unit, clause);

                    for (resolvent, resolved_lit) in resolvents {
                        steps += 1;
                        self.stats.resolution_steps += 1;

                        if resolvent.is_tautology() {
                            self.stats.tautologies_removed += 1;
                            continue;
                        }

                        if resolvent.is_empty() {
                            self.stats.empty_clause_found = true;
                            derivation.push(ResolutionStep {
                                parent1: unit.clone(),
                                parent2: clause.clone(),
                                resolvent: resolvent.clone(),
                                resolved_literal: resolved_lit,
                            });
                            return ProofResult::Unsatisfiable { steps, derivation };
                        }

                        if !clauses.contains(&resolvent) && !new_clauses.contains(&resolvent) {
                            new_clauses.push(resolvent.clone());
                            found_new = true;
                            self.stats.clauses_generated += 1;
                            derivation.push(ResolutionStep {
                                parent1: unit.clone(),
                                parent2: clause.clone(),
                                resolvent,
                                resolved_literal: resolved_lit,
                            });
                        }
                    }
                }
            }

            if !found_new {
                return ProofResult::Satisfiable;
            }

            clauses.extend(new_clauses);
        }
    }

    /// Linear resolution strategy.
    fn prove_linear(&mut self, max_depth: usize) -> ProofResult {
        // Simplified linear resolution from first clause
        if self.clauses.is_empty() {
            return ProofResult::Satisfiable;
        }

        let start = self.clauses[0].clone();
        let mut current = start.clone();
        let mut depth = 0;
        let mut derivation = Vec::new();

        while depth < max_depth {
            if current.is_empty() {
                self.stats.empty_clause_found = true;
                return ProofResult::Unsatisfiable {
                    steps: depth,
                    derivation,
                };
            }

            // Try to resolve with any other clause
            let mut resolved = false;
            for other in &self.clauses[1..] {
                let resolvents = self.resolve(&current, other);

                if let Some((resolvent, resolved_lit)) = resolvents.first() {
                    if !resolvent.is_tautology() {
                        current = resolvent.clone();
                        depth += 1;
                        self.stats.resolution_steps += 1;
                        self.stats.clauses_generated += 1;
                        derivation.push(ResolutionStep {
                            parent1: current.clone(),
                            parent2: other.clone(),
                            resolvent: resolvent.clone(),
                            resolved_literal: resolved_lit.clone(),
                        });
                        resolved = true;
                        break;
                    }
                }
            }

            if !resolved {
                return ProofResult::Satisfiable;
            }
        }

        ProofResult::ResourceLimitReached {
            steps_attempted: depth,
        }
    }
}

impl Default for ResolutionProver {
    fn default() -> Self {
        Self::new()
    }
}
