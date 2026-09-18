//! Regression tests for the symbolic-algebra fixes: common subexpression
//! elimination (exact structural matching, unique temporaries), Sturm sequences
//! of degenerate polynomials and polynomial division by the zero polynomial.

use super::functions::{
    common_subexpression_elimination, count_roots_in_interval, cst, eval_expr, sturm_sequence, var,
};
use super::types::{CseResult, Polynomial};
use std::collections::HashMap;

/// Values of `x`, `y` and `_cse0` for the evaluation helpers.
fn env(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
}

/// Evaluates the bindings of `result` in order, then its reduced expression.
///
/// Every temporary must be new (neither a variable of `env` nor an earlier
/// temporary) and may only use variables and earlier temporaries, which the
/// strict evaluation enforces: an unbound name makes `eval_expr` return `None`.
fn eval_cse(result: &CseResult, env: &HashMap<String, f64>) -> f64 {
    let mut scope = env.clone();
    for (name, definition) in &result.bindings {
        let value = eval_expr(definition, &scope)
            .unwrap_or_else(|| panic!("binding {name} uses an undefined name: {definition}"));
        assert!(
            scope.insert(name.clone(), value).is_none(),
            "temporary {name} is defined twice or shadows a variable"
        );
    }
    eval_expr(&result.reduced, &scope).expect("the reduced expression is defined")
}

// ---------------------------------------------------------------------------
// common_subexpression_elimination
// ---------------------------------------------------------------------------

#[test]
fn cse_keeps_constants_that_print_alike_apart() {
    // Both products used to print as `(x * 0.000000)`: they were merged and the
    // sum became 2 * (1e-7 x) instead of 1e-7 x + 2e-7 x.
    let e = var("x")
        .mul_expr(cst(1e-7))
        .add_expr(var("x").mul_expr(cst(2e-7)));
    let result = common_subexpression_elimination(&e);
    assert!(result.bindings.is_empty());
    assert_eq!(result.reduced, e);
    let at_one = env(&[("x", 1.0)]);
    let value = eval_cse(&result, &at_one);
    assert_eq!(Some(value), eval_expr(&e, &at_one));
    assert!((value - 3e-7).abs() < 1e-20, "value = {value}");
}

#[test]
fn cse_keeps_sums_with_constants_that_agree_to_six_decimals_apart() {
    // 0.1234561 and 0.1234564 both print as 0.123456.
    let e = var("x")
        .add_expr(cst(0.1234561))
        .mul_expr(var("x").add_expr(cst(0.1234564)));
    let result = common_subexpression_elimination(&e);
    assert!(result.bindings.is_empty());
    let at_two = env(&[("x", 2.0)]);
    let expected = (2.0 + 0.1234561) * (2.0 + 0.1234564);
    assert_eq!(eval_cse(&result, &at_two), expected);
    assert_eq!(eval_expr(&e, &at_two), Some(expected));
}

#[test]
fn cse_distinguishes_every_constant_bit_pattern() {
    let e = var("x")
        .add_expr(cst(0.3))
        .mul_expr(var("x").add_expr(cst(0.1 + 0.2)));
    let result = common_subexpression_elimination(&e);
    assert!(
        result.bindings.is_empty(),
        "0.3 and 0.1 + 0.2 differ in the last bit"
    );
    assert_eq!(result.reduced, e);
    // Positive and negative zero are different constants as well.
    let signed_zero = var("x")
        .add_expr(cst(0.0))
        .mul_expr(var("x").add_expr(cst(-0.0)));
    assert!(
        common_subexpression_elimination(&signed_zero)
            .bindings
            .is_empty()
    );
}

#[test]
fn cse_extracts_a_repeated_subexpression_once() {
    let sub = var("x").add_expr(cst(1.0));
    let e = sub
        .clone()
        .mul_expr(sub.clone())
        .add_expr(sub.clone().sin_expr());
    let result = common_subexpression_elimination(&e);
    assert_eq!(result.bindings.len(), 1);
    assert_eq!(result.bindings[0], ("_cse0".to_string(), sub));
    let temp = var("_cse0");
    assert_eq!(
        result.reduced,
        temp.clone()
            .mul_expr(temp.clone())
            .add_expr(temp.sin_expr())
    );
    let at = env(&[("x", 0.7)]);
    assert_eq!(Some(eval_cse(&result, &at)), eval_expr(&e, &at));
}

#[test]
fn cse_names_nested_temporaries_uniquely_and_in_dependency_order() {
    // (x+1) occurs three times and (x+1)*2 twice.  The inner temporary used to
    // be named before the outer one was pushed, so both were `_cse0` and the
    // third (x+1) read the wrong value.
    let sub = var("x").add_expr(cst(1.0));
    let doubled = sub.clone().mul_expr(cst(2.0));
    let e = doubled
        .clone()
        .add_expr(doubled.clone())
        .add_expr(sub.clone());
    let result = common_subexpression_elimination(&e);
    assert_eq!(result.bindings.len(), 2);
    assert_eq!(result.bindings[0], ("_cse0".to_string(), sub));
    assert_eq!(
        result.bindings[1],
        ("_cse1".to_string(), var("_cse0").mul_expr(cst(2.0)))
    );
    assert_eq!(
        result.reduced,
        var("_cse1").add_expr(var("_cse1")).add_expr(var("_cse0"))
    );
    for x in [-2.5, 0.0, 0.5, 3.0] {
        let at = env(&[("x", x)]);
        assert_eq!(Some(eval_cse(&result, &at)), eval_expr(&e, &at), "x = {x}");
    }
}

#[test]
fn cse_does_not_extract_a_subtree_used_only_inside_a_shared_one() {
    // (x+1) appears twice in the tree but only inside the shared (x+1)*2, which
    // is computed once: a temporary for (x+1) alone would be used once.
    let doubled = var("x").add_expr(cst(1.0)).mul_expr(cst(2.0));
    let e = doubled.clone().add_expr(doubled.clone());
    let result = common_subexpression_elimination(&e);
    assert_eq!(result.bindings.len(), 1);
    assert_eq!(result.bindings[0], ("_cse0".to_string(), doubled));
    assert_eq!(result.reduced, var("_cse0").add_expr(var("_cse0")));
}

#[test]
fn cse_temporaries_do_not_shadow_variables_of_the_input() {
    let sub = var("_cse0").add_expr(cst(1.0));
    let e = sub.clone().mul_expr(sub);
    let result = common_subexpression_elimination(&e);
    assert_eq!(result.bindings.len(), 1);
    assert_eq!(result.bindings[0].0, "_cse1");
    let at = env(&[("_cse0", 2.0)]);
    assert_eq!(eval_cse(&result, &at), 9.0);
}

#[test]
fn cse_of_an_expression_without_repeats_is_the_identity() {
    let e = var("x")
        .sin_expr()
        .add_expr(var("y").cos_expr().mul_expr(var("z").exp_expr()))
        .div_expr(cst(2.0).add_expr(var("x").ln_expr()));
    let result = common_subexpression_elimination(&e);
    assert!(result.bindings.is_empty());
    assert_eq!(result.reduced, e);
    assert_eq!(
        common_subexpression_elimination(&cst(5.0)).reduced,
        cst(5.0)
    );
    assert_eq!(
        common_subexpression_elimination(&var("x")).reduced,
        var("x")
    );
}

#[test]
fn cse_binds_each_level_of_a_doubling_tree_once() {
    // e_0 = x, e_{k+1} = e_k + e_k: the tree has 2^15 nodes but only 14 distinct
    // ones, so exactly one temporary per level (the root stays in `reduced`).
    let mut e = var("x");
    for _ in 0..14 {
        e = e.clone().add_expr(e);
    }
    assert!(e.node_count() > 30_000);
    let result = common_subexpression_elimination(&e);
    assert_eq!(result.bindings.len(), 13);
    assert_eq!(
        result.bindings[0],
        ("_cse0".to_string(), var("x").add_expr(var("x")))
    );
    assert_eq!(
        result.bindings[12],
        ("_cse12".to_string(), var("_cse11").add_expr(var("_cse11")))
    );
    assert_eq!(result.reduced, var("_cse12").add_expr(var("_cse12")));
    assert_eq!(eval_cse(&result, &env(&[("x", 1.5)])), 1.5 * 16384.0);
}

#[test]
fn cse_preserves_the_value_of_a_mixed_expression() {
    let xy = var("x").mul_expr(var("y"));
    let e = xy
        .clone()
        .add_expr(xy.clone().sin_expr())
        .mul_expr(xy.clone().exp_expr())
        .div_expr(xy.add_expr(cst(2.0)))
        .add_expr(var("x").neg_expr().pow_expr(cst(2.0)));
    let result = common_subexpression_elimination(&e);
    assert!(!result.bindings.is_empty());
    for (x, y) in [(0.3, 0.9), (-1.2, 0.4), (2.0, -0.25)] {
        let at = env(&[("x", x), ("y", y)]);
        let direct = eval_expr(&e, &at).expect("defined");
        let reduced = eval_cse(&result, &at);
        assert!(
            (direct - reduced).abs() <= 1e-12 * direct.abs().max(1.0),
            "x = {x}, y = {y}: {direct} vs {reduced}"
        );
    }
}

// ---------------------------------------------------------------------------
// sturm_sequence / count_roots_in_interval
// ---------------------------------------------------------------------------

/// The monic polynomial with the given roots.
fn polynomial_from_roots(roots: &[f64]) -> Polynomial {
    roots.iter().fold(Polynomial::constant(1.0), |acc, &r| {
        acc.mul(&Polynomial::new(vec![-r, 1.0]))
    })
}

#[test]
fn sturm_sequence_of_a_constant_polynomial_is_the_polynomial() {
    // derivative() of a constant is the zero polynomial, and dividing by it
    // used to panic.
    let seq = sturm_sequence(&Polynomial::constant(3.0));
    assert_eq!(seq, vec![Polynomial::constant(3.0)]);
    assert_eq!(
        sturm_sequence(&Polynomial::zero()),
        vec![Polynomial::zero()]
    );
}

#[test]
fn constant_polynomials_have_no_roots() {
    assert_eq!(
        count_roots_in_interval(&Polynomial::constant(3.0), 0.0, 1.0),
        0
    );
    assert_eq!(
        count_roots_in_interval(&Polynomial::constant(-1e6), -1e9, 1e9),
        0
    );
    assert_eq!(count_roots_in_interval(&Polynomial::zero(), -1.0, 1.0), 0);
    // A constant stored with trailing zero coefficients.
    let untrimmed = Polynomial {
        coeffs: vec![2.0, 0.0, 0.0],
    };
    assert_eq!(count_roots_in_interval(&untrimmed, -5.0, 5.0), 0);
}

#[test]
fn roots_of_a_polynomial_with_large_coefficients_are_counted() {
    // (x - 1000)(x - 2000) = x^2 - 3000 x + 2e6.
    let p = polynomial_from_roots(&[1000.0, 2000.0]);
    assert_eq!(p.coeffs, vec![2.0e6, -3000.0, 1.0]);
    assert_eq!(count_roots_in_interval(&p, 0.0, 3000.0), 2);
    assert_eq!(count_roots_in_interval(&p, 0.0, 1500.0), 1);
    assert_eq!(count_roots_in_interval(&p, 1500.0, 3000.0), 1);
    assert_eq!(count_roots_in_interval(&p, 2500.0, 3000.0), 0);
    // The interval is (a, b]: a root at b counts, a root at a does not.
    assert_eq!(count_roots_in_interval(&p, 0.0, 2000.0), 2);
    assert_eq!(count_roots_in_interval(&p, 1000.0, 3000.0), 1);
    assert_eq!(count_roots_in_interval(&p, 0.0, 1000.0), 1);
}

#[test]
fn a_sturm_sequence_has_at_most_degree_plus_one_terms() {
    // Rounding residues of the remainders used to add spurious terms to the
    // sequence of a polynomial with large coefficients.
    let roots = [100.5, 200.5, 300.5, 400.5, 500.5];
    let p = polynomial_from_roots(&roots);
    let seq = sturm_sequence(&p);
    assert_eq!(seq.len(), 6);
    assert_eq!(count_roots_in_interval(&p, 0.0, 1000.0), 5);
    assert_eq!(count_roots_in_interval(&p, 150.0, 350.0), 2);
}

#[test]
fn a_repeated_root_is_counted_once_despite_rounding_residue() {
    // (x - 100.1)^2 (x - 700.7)(x - 1300.3): the last remainder is zero in exact
    // arithmetic, but in floating point it is a residue far above 1e-15 that was
    // kept as a spurious Sturm term and inflated the count to four.
    let p = polynomial_from_roots(&[100.1, 100.1, 700.7, 1300.3]);
    let seq = sturm_sequence(&p);
    assert!(seq.len() <= 4, "sequence has {} terms", seq.len());
    assert_eq!(count_roots_in_interval(&p, -10.0, 3000.0), 3);
    assert_eq!(count_roots_in_interval(&p, 0.0, 400.0), 1);
    assert_eq!(count_roots_in_interval(&p, 0.0, 800.0), 2);
    assert_eq!(count_roots_in_interval(&p, 800.0, 2000.0), 1);
}

#[test]
fn sturm_counts_match_the_known_roots_of_ordinary_polynomials() {
    let p = Polynomial::new(vec![-1.0, 0.0, 1.0]);
    assert_eq!(count_roots_in_interval(&p, -2.0, 2.0), 2);
    assert_eq!(count_roots_in_interval(&p, -0.5, 2.0), 1);
    // x^2 + 1 has no real roots.
    let no_real = Polynomial::new(vec![1.0, 0.0, 1.0]);
    assert_eq!(count_roots_in_interval(&no_real, -10.0, 10.0), 0);
    // A linear polynomial has its single root.
    let line = Polynomial::new(vec![-3.0, 2.0]);
    assert_eq!(count_roots_in_interval(&line, 0.0, 2.0), 1);
    assert_eq!(count_roots_in_interval(&line, 2.0, 5.0), 0);
    // (x + 3)(x + 1)(x - 2)(x - 4)
    let quartic = polynomial_from_roots(&[-3.0, -1.0, 2.0, 4.0]);
    assert_eq!(count_roots_in_interval(&quartic, -10.0, 10.0), 4);
    assert_eq!(count_roots_in_interval(&quartic, -2.0, 3.0), 2);
    assert_eq!(count_roots_in_interval(&quartic, 0.0, 1.0), 0);
}

// ---------------------------------------------------------------------------
// Polynomial::div_rem
// ---------------------------------------------------------------------------

#[test]
fn division_by_the_zero_polynomial_is_none() {
    let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
    assert!(p.div_rem(&Polynomial::zero()).is_none());
    assert!(p.div_rem(&Polynomial { coeffs: vec![] }).is_none());
    assert!(
        p.div_rem(&Polynomial {
            coeffs: vec![0.0, 0.0]
        })
        .is_none()
    );
    assert!(
        p.div_rem(&Polynomial {
            coeffs: vec![1e-16]
        })
        .is_none()
    );
}

#[test]
fn division_ignores_trailing_zero_coefficients_of_the_divisor() {
    // x - 1 stored as [-1, 1, 0, 0]: the leading coefficient is not zero.
    let divisor = Polynomial {
        coeffs: vec![-1.0, 1.0, 0.0, 0.0],
    };
    let dividend = Polynomial::new(vec![-1.0, 0.0, 1.0]);
    let (q, r) = dividend.div_rem(&divisor).expect("non-zero divisor");
    assert_eq!(q, Polynomial::new(vec![1.0, 1.0]));
    assert!(r.is_zero());
}

#[test]
fn division_reconstructs_the_dividend() {
    let a = Polynomial::new(vec![3.0, -2.0, 0.5, 4.0, 1.0]);
    let b = Polynomial::new(vec![1.0, -3.0, 2.0]);
    let (q, r) = a.div_rem(&b).expect("non-zero divisor");
    assert!(r.degree() < b.degree());
    let rebuilt = q.mul(&b).add(&r);
    assert_eq!(rebuilt.coeffs.len(), a.coeffs.len());
    for (got, want) in rebuilt.coeffs.iter().zip(&a.coeffs) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    // A dividend of lower degree is its own remainder.
    let (q, r) = b.div_rem(&a).expect("non-zero divisor");
    assert!(q.is_zero());
    assert_eq!(r, b);
}

#[test]
fn division_by_a_constant_leaves_exactly_no_remainder() {
    // x^2 - 3000 x + 2e6 and its derivative 2x - 3000 divided by 250000: the
    // remainder is zero by construction, but the elimination used to leave a
    // rounding residue of about 5e-13 in it.
    let derivative = Polynomial::new(vec![-3000.0, 2.0]);
    let (q, r) = derivative
        .div_rem(&Polynomial::constant(250_000.0))
        .expect("non-zero divisor");
    assert_eq!(r.coeffs, vec![0.0]);
    assert!((q.eval(1.0) - (-2998.0 / 250_000.0)).abs() < 1e-15);
}

#[test]
fn gcd_with_the_zero_polynomial_does_not_panic() {
    let p = Polynomial::new(vec![-1.0, 0.0, 1.0]);
    let g = p.gcd(&Polynomial::zero());
    assert_eq!(g.degree(), 2);
    assert!((g.eval(1.0)).abs() < 1e-12);
    let both_zero = Polynomial::zero().gcd(&Polynomial::zero());
    assert!(both_zero.is_zero());
}
