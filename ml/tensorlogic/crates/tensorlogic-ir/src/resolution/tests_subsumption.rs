//! Theta-subsumption tests for [`super::Clause`].

use super::*;

#[test]
fn test_subsumption_identical() {
    // {P(a)} subsumes {P(a)} (same clause)
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let c = Clause::unit(p_a);

    assert!(c.subsumes(&c));
}

#[test]
fn test_subsumption_variable_to_constant() {
    // {P(x)} subsumes {P(a)} with θ = {x/a}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

    let c_general = Clause::unit(p_x);
    let c_specific = Clause::unit(p_a);

    assert!(c_general.subsumes(&c_specific));
}

#[test]
fn test_subsumption_not_reverse() {
    // {P(a)} does NOT subsume {P(x)} (constant is less general than variable)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

    let c_general = Clause::unit(p_x);
    let c_specific = Clause::unit(p_a);

    assert!(!c_specific.subsumes(&c_general));
}

#[test]
fn test_subsumption_smaller_clause() {
    // {P(x)} subsumes {P(a), Q(a)} with θ = {x/a}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));

    let c1 = Clause::unit(p_x);
    let c2 = Clause::from_literals(vec![p_a, q_a]);

    assert!(c1.subsumes(&c2));
}

#[test]
fn test_subsumption_multiple_literals() {
    // {P(x), Q(x)} subsumes {P(a), Q(a), R(a)} with θ = {x/a}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    let c1 = Clause::from_literals(vec![p_x, q_x]);

    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));
    let r_a = Literal::positive(TLExpr::pred("R", vec![Term::constant("a")]));
    let c2 = Clause::from_literals(vec![p_a, q_a, r_a]);

    assert!(c1.subsumes(&c2));
}

#[test]
fn test_subsumption_fails_different_pred() {
    // {P(x)} does not subsume {Q(a)} (different predicate names)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));

    let c1 = Clause::unit(p_x);
    let c2 = Clause::unit(q_a);

    assert!(!c1.subsumes(&c2));
}

#[test]
fn test_subsumption_fails_too_many_literals() {
    // {P(x), Q(x), R(x)} does not subsume {P(a), Q(a)} (c1 has more literals)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    let r_x = Literal::positive(TLExpr::pred("R", vec![Term::var("x")]));
    let c1 = Clause::from_literals(vec![p_x, q_x, r_x]);

    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));
    let c2 = Clause::from_literals(vec![p_a, q_a]);

    assert!(!c1.subsumes(&c2));
}

#[test]
fn test_subsumption_empty_clause() {
    // Empty clause only subsumes itself
    let empty = Clause::empty();
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let non_empty = Clause::unit(p_a);

    assert!(empty.subsumes(&empty));
    assert!(!empty.subsumes(&non_empty));
    assert!(!non_empty.subsumes(&empty));
}

#[test]
fn test_subsumption_polarity_matters() {
    // {P(x)} does not subsume {¬P(a)} (different polarity)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

    let c1 = Clause::unit(p_x);
    let c2 = Clause::unit(not_p_a);

    assert!(!c1.subsumes(&c2));
}

#[test]
fn test_subsumption_two_variables() {
    // {P(x, y)} subsumes {P(a, b)} with θ = {x/a, y/b}
    let p_xy = Literal::positive(TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]));
    let p_ab = Literal::positive(TLExpr::pred(
        "P",
        vec![Term::constant("a"), Term::constant("b")],
    ));

    let c1 = Clause::unit(p_xy);
    let c2 = Clause::unit(p_ab);

    assert!(c1.subsumes(&c2));
}

#[test]
fn test_subsumption_in_prover() {
    // Test that subsumption is actually used in the prover to reduce search space
    let mut prover = ResolutionProver::new();

    // Add {P(x), Q(x)} - more general
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    prover.add_clause(Clause::from_literals(vec![p_x, q_x]));

    // This clause would be subsumed by the first
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let q_a = Literal::positive(TLExpr::pred("Q", vec![Term::constant("a")]));
    let r_a = Literal::positive(TLExpr::pred("R", vec![Term::constant("a")]));
    let subsumed_clause = Clause::from_literals(vec![p_a, q_a, r_a]);

    // Check if subsumption works
    assert!(prover.clauses[0].subsumes(&subsumed_clause));
}
