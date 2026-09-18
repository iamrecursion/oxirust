//! Unit tests for the resolution prover: basic literal/clause tests,
//! propositional resolution, and CNF conversion.

use super::*;

fn p() -> TLExpr {
    TLExpr::pred("P", vec![])
}

fn q() -> TLExpr {
    TLExpr::pred("Q", vec![])
}

fn r() -> TLExpr {
    TLExpr::pred("R", vec![])
}

#[test]
fn test_literal_creation() {
    let lit_pos = Literal::positive(p());
    assert!(lit_pos.is_positive());
    assert!(!lit_pos.is_negative());

    let lit_neg = Literal::negative(p());
    assert!(!lit_neg.is_positive());
    assert!(lit_neg.is_negative());
}

#[test]
fn test_literal_complementary() {
    let lit_pos = Literal::positive(p());
    let lit_neg = Literal::negative(p());

    assert!(lit_pos.is_complementary(&lit_neg));
    assert!(lit_neg.is_complementary(&lit_pos));
    assert!(!lit_pos.is_complementary(&lit_pos));
}

#[test]
fn test_clause_empty() {
    let clause = Clause::empty();
    assert!(clause.is_empty());
    assert_eq!(clause.len(), 0);
}

#[test]
fn test_clause_unit() {
    let clause = Clause::unit(Literal::positive(p()));
    assert!(clause.is_unit());
    assert_eq!(clause.len(), 1);
}

#[test]
fn test_clause_tautology() {
    // P ∨ ¬P is a tautology
    let clause = Clause::from_literals(vec![Literal::positive(p()), Literal::negative(p())]);
    assert!(clause.is_tautology());
}

#[test]
fn test_resolution_basic() {
    // {P}, {¬P} ⊢ ∅
    let mut prover = ResolutionProver::new();
    prover.add_clause(Clause::unit(Literal::positive(p())));
    prover.add_clause(Clause::unit(Literal::negative(p())));

    let result = prover.prove();
    assert!(result.is_unsatisfiable());
}

#[test]
fn test_resolution_modus_ponens() {
    // {P}, {P → Q} ≡ {P}, {¬P ∨ Q} ⊢ Q
    // Clauses: {P}, {¬P, Q}
    // Resolution: {Q}
    let mut prover = ResolutionProver::new();
    prover.add_clause(Clause::unit(Literal::positive(p())));
    prover.add_clause(Clause::from_literals(vec![
        Literal::negative(p()),
        Literal::positive(q()),
    ]));
    // To prove Q, add ¬Q and check for contradiction
    prover.add_clause(Clause::unit(Literal::negative(q())));

    let result = prover.prove();
    assert!(result.is_unsatisfiable());
}

#[test]
fn test_resolution_satisfiable() {
    // {P}, {Q} is satisfiable (no complementary literals)
    let mut prover = ResolutionProver::new();
    prover.add_clause(Clause::unit(Literal::positive(p())));
    prover.add_clause(Clause::unit(Literal::positive(q())));

    let result = prover.prove();
    // Should saturate or be satisfiable
    assert!(!result.is_unsatisfiable());
}

#[test]
fn test_cnf_conversion_and() {
    // P ∧ Q → clauses: {P}, {Q}
    let expr = TLExpr::and(p(), q());
    let clauses = to_cnf(&expr).expect("unwrap");

    assert_eq!(clauses.len(), 2);
    assert!(clauses.iter().all(|c| c.is_unit()));
}

#[test]
fn test_cnf_conversion_or() {
    // P ∨ Q → clause: {P, Q}
    let expr = TLExpr::or(p(), q());
    let clauses = to_cnf(&expr).expect("unwrap");

    assert_eq!(clauses.len(), 1);
    assert_eq!(clauses[0].len(), 2);
}

#[test]
fn test_resolution_strategy_unit() {
    // Test unit resolution strategy
    let mut prover =
        ResolutionProver::with_strategy(ResolutionStrategy::UnitResolution { max_steps: 100 });

    prover.add_clause(Clause::unit(Literal::positive(p())));
    prover.add_clause(Clause::unit(Literal::negative(p())));

    let result = prover.prove();
    assert!(result.is_unsatisfiable());
}

#[test]
fn test_resolution_three_clauses() {
    // {P ∨ Q}, {¬P ∨ R}, {¬Q}, {¬R} ⊢ ∅
    let mut prover = ResolutionProver::new();

    prover.add_clause(Clause::from_literals(vec![
        Literal::positive(p()),
        Literal::positive(q()),
    ]));
    prover.add_clause(Clause::from_literals(vec![
        Literal::negative(p()),
        Literal::positive(r()),
    ]));
    prover.add_clause(Clause::unit(Literal::negative(q())));
    prover.add_clause(Clause::unit(Literal::negative(r())));

    let result = prover.prove();
    assert!(result.is_unsatisfiable());
}

#[test]
fn test_horn_clause_detection() {
    // {¬P, ¬Q, R} is a Horn clause (exactly one positive)
    let clause = Clause::from_literals(vec![
        Literal::negative(p()),
        Literal::negative(q()),
        Literal::positive(r()),
    ]);
    assert!(clause.is_horn());

    // {P, Q} is not a Horn clause (two positives)
    let non_horn = Clause::from_literals(vec![Literal::positive(p()), Literal::positive(q())]);
    assert!(!non_horn.is_horn());
}

#[test]
fn test_prover_stats() {
    let mut prover = ResolutionProver::new();
    prover.add_clause(Clause::unit(Literal::positive(p())));
    prover.add_clause(Clause::unit(Literal::negative(p())));

    let result = prover.prove();

    assert!(prover.stats.empty_clause_found);
    assert!(prover.stats.resolution_steps > 0);
    assert!(result.is_unsatisfiable());
}
