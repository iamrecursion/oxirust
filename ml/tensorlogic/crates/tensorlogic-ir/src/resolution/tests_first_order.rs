//! First-order resolution tests (unification, standardizing-apart, and
//! substitution application on literals and clauses).

use super::*;

#[test]
fn test_literal_unification_ground() {
    // P(a) and ¬P(a) should unify with empty substitution
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

    let mgu = p_a.try_unify(&not_p_a);
    assert!(mgu.is_some());
    assert!(mgu.expect("unwrap").is_empty());
}

#[test]
fn test_literal_unification_variable() {
    // P(x) and ¬P(a) should unify with {x/a}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

    let mgu = p_x.try_unify(&not_p_a);
    assert!(mgu.is_some());

    let mgu = mgu.expect("unwrap");
    assert_eq!(mgu.len(), 1);
    assert_eq!(mgu.apply(&Term::var("x")), Term::constant("a"));
}

#[test]
fn test_literal_unification_fails_diff_names() {
    // P(x) and ¬Q(x) should not unify (different predicate names)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let not_q_x = Literal::negative(TLExpr::pred("Q", vec![Term::var("x")]));

    let mgu = p_x.try_unify(&not_q_x);
    assert!(mgu.is_none());
}

#[test]
fn test_literal_unification_fails_same_polarity() {
    // P(x) and P(a) should not unify (same polarity)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

    let mgu = p_x.try_unify(&p_a);
    assert!(mgu.is_none());
}

#[test]
fn test_literal_apply_substitution() {
    // P(x) with {x/a} should become P(a)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let mut subst = Substitution::empty();
    subst.bind("x".to_string(), Term::constant("a"));

    let p_a = p_x.apply_substitution(&subst);
    let expected = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

    assert_eq!(p_a.atom, expected.atom);
    assert_eq!(p_a.polarity, expected.polarity);
}

#[test]
fn test_clause_rename_variables() {
    // P(x) ∨ Q(x) renamed with "1" should become P(x_1) ∨ Q(x_1)
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    let clause = Clause::from_literals(vec![p_x, q_x]);

    let renamed = clause.rename_variables("1");

    // Check that variables were renamed
    let vars = renamed.free_vars();
    assert!(vars.contains("x_1"));
    assert!(!vars.contains("x"));
}

#[test]
fn test_clause_apply_substitution() {
    // {P(x), Q(y)} with {x/a, y/b} should become {P(a), Q(b)}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_y = Literal::positive(TLExpr::pred("Q", vec![Term::var("y")]));
    let clause = Clause::from_literals(vec![p_x, q_y]);

    let mut subst = Substitution::empty();
    subst.bind("x".to_string(), Term::constant("a"));
    subst.bind("y".to_string(), Term::constant("b"));

    let result = clause.apply_substitution(&subst);

    // Should have no free variables (all substituted)
    assert!(result.free_vars().is_empty());
}

#[test]
fn test_first_order_resolution_basic() {
    // {P(x)} and {¬P(a)} should resolve to {} with {x/a}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

    let c1 = Clause::unit(p_x);
    let c2 = Clause::unit(not_p_a);

    let prover = ResolutionProver::new();
    let resolvents = prover.resolve_first_order(&c1, &c2);

    assert_eq!(resolvents.len(), 1);
    assert!(resolvents[0].0.is_empty()); // Empty clause derived
}

#[test]
fn test_first_order_resolution_complex() {
    // {P(x), Q(x)} and {¬P(a), R(a)} should resolve to {Q(a), R(a)}
    let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let q_x = Literal::positive(TLExpr::pred("Q", vec![Term::var("x")]));
    let c1 = Clause::from_literals(vec![p_x, q_x]);

    let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
    let r_a = Literal::positive(TLExpr::pred("R", vec![Term::constant("a")]));
    let c2 = Clause::from_literals(vec![not_p_a, r_a]);

    let prover = ResolutionProver::new();
    let resolvents = prover.resolve_first_order(&c1, &c2);

    assert_eq!(resolvents.len(), 1);
    let resolvent = &resolvents[0].0;

    // Should have 2 literals: Q(a) and R(a)
    assert_eq!(resolvent.len(), 2);

    // Should have no free variables (all unified)
    assert!(resolvent.free_vars().is_empty());
}

#[test]
fn test_first_order_resolution_multiple_vars() {
    // {P(x, y)} and {¬P(a, b)} should resolve to {} with {x/a, y/b}
    let p_xy = Literal::positive(TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]));
    let not_p_ab = Literal::negative(TLExpr::pred(
        "P",
        vec![Term::constant("a"), Term::constant("b")],
    ));

    let c1 = Clause::unit(p_xy);
    let c2 = Clause::unit(not_p_ab);

    let prover = ResolutionProver::new();
    let resolvents = prover.resolve_first_order(&c1, &c2);

    assert_eq!(resolvents.len(), 1);
    assert!(resolvents[0].0.is_empty());
}

#[test]
fn test_first_order_resolution_standardizing_apart() {
    // {P(x)} and {¬P(x)} should be standardized apart before resolution
    // After standardization: {P(x_c1_N)} and {¬P(x_c2_N)}
    // These should resolve to {} with {x_c1_N/x_c2_N} or similar
    let p_x1 = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));
    let not_p_x2 = Literal::negative(TLExpr::pred("P", vec![Term::var("x")]));

    let c1 = Clause::unit(p_x1);
    let c2 = Clause::unit(not_p_x2);

    let prover = ResolutionProver::new();
    let resolvents = prover.resolve_first_order(&c1, &c2);

    // Should successfully resolve despite both using variable "x"
    assert_eq!(resolvents.len(), 1);
    assert!(resolvents[0].0.is_empty());
}

#[test]
fn test_first_order_resolution_no_unifier() {
    // {P(a)} and {¬P(b)} should not resolve (no unifier for a and b)
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
    let not_p_b = Literal::negative(TLExpr::pred("P", vec![Term::constant("b")]));

    let c1 = Clause::unit(p_a);
    let c2 = Clause::unit(not_p_b);

    let prover = ResolutionProver::new();
    let resolvents = prover.resolve_first_order(&c1, &c2);

    // Should find no resolvents
    assert_eq!(resolvents.len(), 0);
}
