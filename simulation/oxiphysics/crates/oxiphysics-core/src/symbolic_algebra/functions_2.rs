//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::symbolic_algebra::Expr;
    use crate::symbolic_algebra::PartialFractionTerm;
    use crate::symbolic_algebra::Polynomial;
    use crate::symbolic_algebra::RationalFunction;
    use std::collections::HashMap;
    #[test]
    fn test_const_eval() {
        let e = cst(42.0);
        let env = HashMap::new();
        assert!((eval_expr(&e, &env).unwrap() - 42.0).abs() < 1e-15);
    }
    #[test]
    fn test_var_eval() {
        let e = var("x");
        let mut env = HashMap::new();
        env.insert("x".into(), 7.0);
        assert!((eval_expr(&e, &env).unwrap() - 7.0).abs() < 1e-15);
    }
    #[test]
    fn test_add_eval() {
        let e = cst(3.0).add_expr(cst(4.0));
        let env = HashMap::new();
        assert!((eval_expr(&e, &env).unwrap() - 7.0).abs() < 1e-15);
    }
    #[test]
    fn test_mul_eval() {
        let e = cst(3.0).mul_expr(cst(5.0));
        let env = HashMap::new();
        assert!((eval_expr(&e, &env).unwrap() - 15.0).abs() < 1e-15);
    }
    #[test]
    fn test_div_eval() {
        let e = cst(10.0).div_expr(cst(4.0));
        let env = HashMap::new();
        assert!((eval_expr(&e, &env).unwrap() - 2.5).abs() < 1e-15);
    }
    #[test]
    fn test_div_by_zero() {
        let e = cst(1.0).div_expr(cst(0.0));
        let env = HashMap::new();
        assert!(eval_expr(&e, &env).is_none());
    }
    #[test]
    fn test_pow_eval() {
        let e = cst(2.0).pow_expr(cst(10.0));
        let env = HashMap::new();
        assert!((eval_expr(&e, &env).unwrap() - 1024.0).abs() < 1e-10);
    }
    #[test]
    fn test_diff_const() {
        let d = diff(&cst(5.0), "x");
        let env = HashMap::new();
        assert!((eval_expr(&simplify(&d), &env).unwrap()).abs() < 1e-15);
    }
    #[test]
    fn test_diff_var() {
        let d = diff(&var("x"), "x");
        let env = HashMap::new();
        assert!((eval_expr(&simplify(&d), &env).unwrap() - 1.0).abs() < 1e-15);
    }
    #[test]
    fn test_diff_x_squared() {
        let e = var("x").pow_expr(cst(2.0));
        let d = simplify(&diff(&e, "x"));
        let mut env = HashMap::new();
        env.insert("x".into(), 3.0);
        assert!((eval_expr(&d, &env).unwrap() - 6.0).abs() < 1e-10);
    }
    #[test]
    fn test_diff_product_rule() {
        let e = var("x").mul_expr(var("x").sin_expr());
        let d = simplify(&diff(&e, "x"));
        let mut env = HashMap::new();
        env.insert("x".into(), 1.0);
        let expected = 1.0_f64.sin() + 1.0_f64.cos();
        assert!((eval_expr(&d, &env).unwrap() - expected).abs() < 1e-8);
    }
    #[test]
    fn test_simplify_zero_add() {
        let e = cst(0.0).add_expr(var("x"));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }
    #[test]
    fn test_simplify_one_mul() {
        let e = cst(1.0).mul_expr(var("x"));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }
    #[test]
    fn test_simplify_zero_mul() {
        let e = cst(0.0).mul_expr(var("x"));
        let s = simplify(&e);
        assert!(s.is_zero());
    }
    #[test]
    fn test_simplify_pow_one() {
        let e = var("x").pow_expr(cst(1.0));
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }
    #[test]
    fn test_simplify_exp_ln() {
        let e = var("x").ln_expr().exp_expr();
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }
    #[test]
    fn test_polynomial_eval() {
        let p = Polynomial::new(vec![2.0, 3.0, 1.0]);
        assert!((p.eval(2.0) - 12.0).abs() < 1e-12);
    }
    #[test]
    fn test_polynomial_add() {
        let a = Polynomial::new(vec![1.0, 2.0]);
        let b = Polynomial::new(vec![3.0, 4.0, 5.0]);
        let c = a.add(&b);
        assert!((c.eval(1.0) - 15.0).abs() < 1e-12);
    }
    #[test]
    fn test_polynomial_mul() {
        let a = Polynomial::new(vec![1.0, 1.0]);
        let b = Polynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert!((c.eval(2.0) - 9.0).abs() < 1e-12);
    }
    #[test]
    fn test_polynomial_derivative() {
        let p = Polynomial::new(vec![1.0, 0.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.eval(2.0) - 12.0).abs() < 1e-12);
    }
    #[test]
    fn test_polynomial_integral() {
        let p = Polynomial::new(vec![6.0]);
        let ip = p.integral();
        assert!((ip.eval(3.0) - 18.0).abs() < 1e-12);
    }
    #[test]
    fn test_polynomial_div_rem() {
        let a = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        let b = Polynomial::new(vec![-1.0, 1.0]);
        let (q, r) = a.div_rem(&b).expect("divisor is non-zero");
        assert!((q.eval(3.0) - 4.0).abs() < 1e-10);
        assert!(r.is_zero());
    }
    #[test]
    fn test_polynomial_gcd() {
        let a = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        let b = Polynomial::new(vec![-1.0, 1.0]);
        let g = a.gcd(&b);
        assert!((g.eval(1.0)).abs() < 1e-10);
        assert!(g.degree() == 1);
    }
    #[test]
    fn test_taylor_exp() {
        let e = var("x").exp_expr();
        let t = taylor_expand(&e, "x", 0.0, 5).expect("exp is analytic at 0");
        assert!((t.eval(1.0) - std::f64::consts::E).abs() < 0.01);
    }
    #[test]
    fn test_taylor_sin() {
        let e = var("x").sin_expr();
        let t = taylor_expand(&e, "x", 0.0, 7).expect("sin is analytic at 0");
        assert!((t.eval(0.5) - 0.5_f64.sin()).abs() < 1e-6);
    }
    #[test]
    fn test_partial_fractions() {
        let num = Polynomial::constant(1.0);
        let roots = vec![1.0, 2.0];
        let terms = partial_fraction_decompose(&num, &roots);
        assert_eq!(terms.len(), 2);
        let val: f64 = terms.iter().map(|t| t.coefficient / (3.0 - t.root)).sum();
        assert!((val - 0.5).abs() < 1e-10);
    }
    #[test]
    fn test_cse_basic() {
        let sub = var("x").add_expr(cst(1.0));
        let e = sub.clone().mul_expr(sub);
        let result = common_subexpression_elimination(&e);
        assert!(!result.bindings.is_empty());
    }
    #[test]
    fn test_substitute() {
        let e = var("x").add_expr(var("y"));
        let s = substitute(&e, "x", &cst(5.0));
        let mut env = HashMap::new();
        env.insert("y".into(), 3.0);
        assert!((eval_expr(&s, &env).unwrap() - 8.0).abs() < 1e-15);
    }
    #[test]
    fn test_substitute_many() {
        let e = var("x").mul_expr(var("y"));
        let mut subs = HashMap::new();
        subs.insert("x".into(), cst(2.0));
        subs.insert("y".into(), cst(3.0));
        let s = substitute_many(&e, &subs);
        let env = HashMap::new();
        assert!((eval_expr(&simplify(&s), &env).unwrap() - 6.0).abs() < 1e-15);
    }
    #[test]
    fn test_display_const() {
        assert_eq!(format!("{}", cst(3.0)), "3");
    }
    #[test]
    fn test_display_add() {
        let e = var("x").add_expr(cst(1.0));
        assert_eq!(format!("{}", e), "(x + 1)");
    }
    #[test]
    fn test_variables() {
        let e = var("x").add_expr(var("y").mul_expr(var("x")));
        let v = e.variables();
        assert_eq!(v, vec!["x", "y"]);
    }
    #[test]
    fn test_node_count() {
        let e = var("x").add_expr(cst(1.0));
        assert_eq!(e.node_count(), 3);
    }
    #[test]
    fn test_depth() {
        let e = var("x").add_expr(cst(1.0));
        assert_eq!(e.depth(), 2);
    }
    #[test]
    fn test_diff_n() {
        let e = var("x").pow_expr(cst(3.0));
        let d2 = diff_n(&e, "x", 2);
        let mut env = HashMap::new();
        env.insert("x".into(), 2.0);
        assert!((eval_expr(&d2, &env).unwrap() - 12.0).abs() < 1e-8);
    }
    #[test]
    fn test_gradient() {
        let f = var("x")
            .pow_expr(cst(2.0))
            .add_expr(var("y").pow_expr(cst(2.0)));
        let g = gradient(&f, &["x", "y"]);
        let mut env = HashMap::new();
        env.insert("x".into(), 3.0);
        env.insert("y".into(), 4.0);
        assert!((eval_expr(&g[0], &env).unwrap() - 6.0).abs() < 1e-8);
        assert!((eval_expr(&g[1], &env).unwrap() - 8.0).abs() < 1e-8);
    }
    #[test]
    fn test_hessian() {
        let f = var("x").pow_expr(cst(2.0)).mul_expr(var("y"));
        let h = hessian(&f, &["x", "y"]);
        let mut env = HashMap::new();
        env.insert("x".into(), 1.0);
        env.insert("y".into(), 2.0);
        assert!((eval_expr(&h[0][0], &env).unwrap() - 4.0).abs() < 1e-8);
    }
    #[test]
    fn test_jacobian() {
        let f1 = var("x").mul_expr(var("y"));
        let f2 = var("x").add_expr(var("y"));
        let j = jacobian(&[f1, f2], &["x", "y"]);
        let mut env = HashMap::new();
        env.insert("x".into(), 2.0);
        env.insert("y".into(), 3.0);
        assert!((eval_expr(&j[0][0], &env).unwrap() - 3.0).abs() < 1e-8);
        assert!((eval_expr(&j[1][1], &env).unwrap() - 1.0).abs() < 1e-8);
    }
    #[test]
    fn test_gradient_check() {
        let e = var("x").pow_expr(cst(3.0));
        let err = gradient_check(&e, "x", &[1.0, 2.0, 3.0], 1e-6);
        assert!(err < 1e-4);
    }
    #[test]
    fn test_rational_function_eval() {
        let rf = RationalFunction::new(
            Polynomial::new(vec![1.0, 1.0]),
            Polynomial::new(vec![-1.0, 1.0]),
        );
        assert!((rf.eval(2.0).unwrap() - 3.0).abs() < 1e-12);
    }
    #[test]
    fn test_rational_function_add() {
        let r1 = RationalFunction::new(Polynomial::constant(1.0), Polynomial::x());
        let r2 = RationalFunction::new(Polynomial::constant(1.0), Polynomial::x());
        let r3 = r1.add(&r2);
        assert!((r3.eval(3.0).unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }
    #[test]
    fn test_lagrange_interpolation() {
        let p = lagrange_interpolation(&[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)]);
        assert!((p.eval(3.0) - 13.0).abs() < 1e-10);
    }
    #[test]
    fn test_chebyshev_t() {
        let t2 = chebyshev_t(2);
        assert!((t2.eval(0.5) - (-0.5)).abs() < 1e-12);
    }
    #[test]
    fn test_hermite_he() {
        let h2 = hermite_he(2);
        assert!((h2.eval(2.0) - 3.0).abs() < 1e-12);
    }
    #[test]
    fn test_legendre_p() {
        let p2 = legendre_p(2);
        assert!((p2.eval(1.0) - 1.0).abs() < 1e-12);
    }
    #[test]
    fn test_find_root_newton() {
        let p = Polynomial::new(vec![-4.0, 0.0, 1.0]);
        let r = find_root_newton(&p, 3.0, 1e-12, 100).unwrap();
        assert!((r - 2.0).abs() < 1e-10);
    }
    #[test]
    fn test_polynomial_composition() {
        let p = Polynomial::new(vec![1.0, 1.0]);
        let q = Polynomial::new(vec![0.0, 2.0]);
        let c = p.compose(&q);
        assert!((c.eval(3.0) - 7.0).abs() < 1e-12);
    }
    #[test]
    fn test_sturm_root_count() {
        let p = Polynomial::new(vec![-1.0, 0.0, 1.0]);
        assert_eq!(count_roots_in_interval(&p, -2.0, 2.0), 2);
    }
    #[test]
    fn test_expand_product() {
        let e = var("x")
            .add_expr(cst(1.0))
            .mul_expr(var("x").add_expr(cst(2.0)));
        let ex = expand(&e);
        let mut env = HashMap::new();
        env.insert("x".into(), 5.0);
        assert!((eval_expr(&simplify(&ex), &env).unwrap() - 42.0).abs() < 1e-10);
    }
    #[test]
    fn test_is_polynomial_expr() {
        let e = var("x").pow_expr(cst(2.0)).add_expr(cst(1.0));
        assert!(is_polynomial_expr(&e));
        let e2 = var("x").sin_expr();
        assert!(!is_polynomial_expr(&e2));
    }
    #[test]
    fn test_polynomial_display() {
        let p = Polynomial::new(vec![1.0, 0.0, 3.0]);
        let s = format!("{}", p);
        assert!(s.contains("x^2"));
    }
    #[test]
    fn test_polynomial_sign_changes() {
        let p = Polynomial::new(vec![1.0, -1.0, 1.0]);
        assert_eq!(p.sign_changes(), 2);
    }
    #[test]
    fn test_expr_to_polynomial() {
        let e = var("x")
            .pow_expr(cst(2.0))
            .add_expr(cst(3.0).mul_expr(var("x")))
            .add_expr(cst(2.0));
        let p = expr_to_polynomial(&e, "x").unwrap();
        assert!((p.eval(1.0) - 6.0).abs() < 1e-12);
    }
    #[test]
    fn test_collect_terms() {
        let e = var("x").add_expr(var("x")).add_expr(cst(1.0));
        let c = collect_terms(&e, "x");
        let mut env = HashMap::new();
        env.insert("x".into(), 5.0);
        assert!((eval_expr(&c, &env).unwrap() - 11.0).abs() < 1e-10);
    }
    #[test]
    fn test_map_expr() {
        let e = var("x").add_expr(cst(5.0));
        let mapped = map_expr(&e, &|ex| match ex {
            Expr::Const(_) => cst(0.0),
            other => other.clone(),
        });
        let mut env = HashMap::new();
        env.insert("x".into(), 3.0);
        assert!((eval_expr(&mapped, &env).unwrap() - 3.0).abs() < 1e-15);
    }
    #[test]
    fn test_simplify_double_neg() {
        let e = var("x").neg_expr().neg_expr();
        let s = simplify(&e);
        assert_eq!(s, var("x"));
    }
    #[test]
    fn test_polynomial_leading_coeff() {
        let p = Polynomial::new(vec![1.0, 2.0, 5.0]);
        assert!((p.leading_coeff() - 5.0).abs() < 1e-15);
    }
    #[test]
    fn test_partial_fractions_to_expr() {
        let terms = vec![
            PartialFractionTerm {
                coefficient: -1.0,
                root: 1.0,
            },
            PartialFractionTerm {
                coefficient: 1.0,
                root: 2.0,
            },
        ];
        let e = partial_fractions_to_expr(&terms, "x");
        let mut env = HashMap::new();
        env.insert("x".into(), 3.0);
        assert!((eval_expr(&e, &env).unwrap() - 0.5).abs() < 1e-12);
    }

    // ── power-rule differentiation and Taylor coefficients (F0927) ────────────

    /// Evaluate `e` with `name = x`, failing the test when it is undefined there.
    fn eval_at(e: &Expr, name: &str, x: f64) -> f64 {
        let mut env = HashMap::new();
        env.insert(name.to_string(), x);
        eval_expr(e, &env).expect("expression must be defined at the probe point")
    }

    /// Coefficient `k` of a polynomial whose trailing zeros may be trimmed.
    fn coeff(p: &Polynomial, k: usize) -> f64 {
        p.coeffs.get(k).copied().unwrap_or(0.0)
    }

    #[test]
    fn test_diff_pow_constant_exponent_is_defined_at_zero() {
        // d(x^2)/dx = 2x.  The old `f^g (g f'/f)` form divided by x and was
        // undefined at x = 0.
        let d = simplify(&diff(&var("x").pow_expr(cst(2.0)), "x"));
        assert_eq!(d, cst(2.0).mul_expr(var("x")));
        assert!(eval_at(&d, "x", 0.0).abs() < 1e-15);
        assert!((eval_at(&d, "x", 3.0) - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_diff_pow_unit_and_zero_exponents() {
        let d1 = simplify(&diff(&var("x").pow_expr(cst(1.0)), "x"));
        assert_eq!(d1, cst(1.0));
        let d0 = simplify(&diff(&var("x").pow_expr(cst(0.0)), "x"));
        assert!(d0.is_zero());
    }

    #[test]
    fn test_diff_pow_negative_and_fractional_exponents() {
        let d = simplify(&diff(&var("x").pow_expr(cst(-1.0)), "x"));
        assert!((eval_at(&d, "x", 2.0) + 0.25).abs() < 1e-14);
        let d = simplify(&diff(&var("x").pow_expr(cst(0.5)), "x"));
        assert!((eval_at(&d, "x", 4.0) - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_diff_pow_exponential_rule() {
        // d(2^x)/dx = 2^x ln 2.
        let d = simplify(&diff(&cst(2.0).pow_expr(var("x")), "x"));
        assert!((eval_at(&d, "x", 3.0) - 8.0 * std::f64::consts::LN_2).abs() < 1e-13);
        assert!((eval_at(&d, "x", 0.0) - std::f64::consts::LN_2).abs() < 1e-14);
    }

    #[test]
    fn test_diff_pow_general_rule_x_to_the_x() {
        // d(x^x)/dx = x^x (ln x + 1).
        let d = simplify(&diff(&var("x").pow_expr(var("x")), "x"));
        let x = 2.0_f64;
        let expected = x.powf(x) * (x.ln() + 1.0);
        assert!((eval_at(&d, "x", x) - expected).abs() < 1e-13);
        assert!((eval_at(&d, "x", 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_diff_pow_with_a_second_variable() {
        let e = var("x").pow_expr(var("y"));
        let mut env = HashMap::new();
        env.insert("x".to_string(), 0.0);
        env.insert("y".to_string(), 2.0);
        // d(x^y)/dx = y x^(y-1) is 0 (not undefined) at x = 0, y = 2.
        let dx = simplify(&diff(&e, "x"));
        assert!(eval_expr(&dx, &env).expect("defined at x = 0").abs() < 1e-15);
        // d(x^y)/dy = x^y ln x
        env.insert("x".to_string(), 2.0);
        env.insert("y".to_string(), 3.0);
        let dy = simplify(&diff(&e, "y"));
        let got = eval_expr(&dy, &env).expect("defined at x = 2");
        assert!((got - 8.0 * std::f64::consts::LN_2).abs() < 1e-13);
    }

    #[test]
    fn test_diff_pow_of_an_independent_expression_is_zero() {
        let e = var("y")
            .pow_expr(cst(2.0))
            .add_expr(cst(3.0).pow_expr(cst(2.0)));
        assert!(simplify(&diff(&e, "x")).is_zero());
    }

    #[test]
    fn test_diff_pow_chain_rule() {
        // d((x^2 + 1)^3)/dx = 3 (x^2 + 1)^2 * 2x
        let inner = var("x").pow_expr(cst(2.0)).add_expr(cst(1.0));
        let d = simplify(&diff(&inner.pow_expr(cst(3.0)), "x"));
        let x = 1.5_f64;
        let expected = 3.0 * (x * x + 1.0).powi(2) * 2.0 * x;
        assert!((eval_at(&d, "x", x) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_diff_pow_matches_central_differences() {
        let x = var("x");
        let cases: Vec<(Expr, Vec<f64>)> = vec![
            (
                x.clone().pow_expr(cst(3.0)),
                vec![-2.0, -0.5, 0.0, 1.0, 2.5],
            ),
            (x.clone().pow_expr(cst(-2.0)), vec![0.5, 1.0, 3.0]),
            (x.clone().pow_expr(cst(0.5)), vec![0.25, 1.0, 9.0]),
            (cst(2.0).pow_expr(x.clone()), vec![-1.0, 0.0, 2.0]),
            (x.clone().pow_expr(x.clone()), vec![0.5, 1.0, 2.5]),
            (x.clone().sin_expr().pow_expr(cst(2.0)), vec![0.3, 1.0, 2.0]),
            (
                cst(3.0).mul_expr(x.clone()).pow_expr(cst(4.0)),
                vec![-1.0, 0.2, 1.5],
            ),
            (
                x.clone().add_expr(cst(2.0)).pow_expr(x.clone().cos_expr()),
                vec![0.1, 1.0, 2.0],
            ),
        ];
        for (e, points) in &cases {
            let err = gradient_check(e, "x", points, 1e-5);
            assert!(err < 1e-6, "{e}: symbolic vs finite-difference error {err}");
        }
    }

    #[test]
    fn test_diff_n_second_derivative_of_x_squared_at_zero() {
        let d2 = diff_n(&var("x").pow_expr(cst(2.0)), "x", 2);
        assert_eq!(d2, cst(2.0));
    }

    #[test]
    fn test_taylor_expand_x_squared_keeps_the_quadratic_coefficient() {
        // Before the fix the second derivative of x^2 at 0 was `None` and was
        // silently replaced by 0.0, giving [0, 0, 0].
        let p =
            taylor_expand(&var("x").pow_expr(cst(2.0)), "x", 0.0, 3).expect("x^2 is analytic at 0");
        let got: Vec<f64> = (0..=3).map(|k| coeff(&p, k)).collect();
        assert_eq!(got, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_taylor_expand_binomial_about_zero() {
        // (1 + x)^3 = 1 + 3x + 3x^2 + x^3; the expansion terminates.
        let e = cst(1.0).add_expr(var("x")).pow_expr(cst(3.0));
        let p = taylor_expand(&e, "x", 0.0, 5).expect("polynomial");
        let got: Vec<f64> = (0..=5).map(|k| coeff(&p, k)).collect();
        for (g, want) in got.iter().zip([1.0, 3.0, 3.0, 1.0, 0.0, 0.0]) {
            assert!((g - want).abs() < 1e-14, "{got:?}");
        }
    }

    #[test]
    fn test_taylor_expand_sqrt_about_four() {
        // sqrt(x) = 2 + (x-4)/4 - (x-4)^2/64 + (x-4)^3/512 - ...
        let e = var("x").pow_expr(cst(0.5));
        let p = taylor_expand(&e, "x", 4.0, 3).expect("analytic at 4");
        let want = [2.0, 0.25, -1.0 / 64.0, 1.0 / 512.0];
        for (k, w) in want.iter().enumerate() {
            assert!(
                (coeff(&p, k) - w).abs() < 1e-14,
                "k = {k}: {}",
                coeff(&p, k)
            );
        }
    }

    #[test]
    fn test_taylor_expand_sin_cos_coefficients() {
        let p = taylor_expand(&var("x").sin_expr(), "x", 0.0, 5).expect("sin");
        let want_sin = [0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0];
        for (k, w) in want_sin.iter().enumerate() {
            assert!((coeff(&p, k) - w).abs() < 1e-14, "sin k = {k}");
        }
        let p = taylor_expand(&var("x").cos_expr(), "x", 0.0, 4).expect("cos");
        let want_cos = [1.0, 0.0, -0.5, 0.0, 1.0 / 24.0];
        for (k, w) in want_cos.iter().enumerate() {
            assert!((coeff(&p, k) - w).abs() < 1e-14, "cos k = {k}");
        }
    }

    #[test]
    fn test_taylor_expand_tangent_through_a_quotient() {
        // tan x = x + x^3 / 3 + ...
        let e = var("x").sin_expr().div_expr(var("x").cos_expr());
        let p = taylor_expand(&e, "x", 0.0, 3).expect("tan is analytic at 0");
        let want = [0.0, 1.0, 0.0, 1.0 / 3.0];
        for (k, w) in want.iter().enumerate() {
            assert!(
                (coeff(&p, k) - w).abs() < 1e-13,
                "k = {k}: {}",
                coeff(&p, k)
            );
        }
    }

    #[test]
    fn test_taylor_expand_off_the_origin_matches_the_function() {
        // exp(2x) about x0 = 0.5: f^(k)(x0) = 2^k e^1, so c_k = 2^k e / k!.
        let e = cst(2.0).mul_expr(var("x")).exp_expr();
        let p = taylor_expand(&e, "x", 0.5, 4).expect("exp is analytic");
        let mut fact = 1.0;
        for k in 0..=4usize {
            if k > 0 {
                fact *= k as f64;
            }
            let want = 2.0_f64.powi(k as i32) * std::f64::consts::E / fact;
            assert!((coeff(&p, k) - want).abs() < 1e-12, "k = {k}");
        }
    }

    #[test]
    fn test_taylor_expand_is_none_when_a_coefficient_is_undefined() {
        // 1/x: the value itself is a pole at 0.
        let pole = cst(1.0).div_expr(var("x"));
        assert!(taylor_expand(&pole, "x", 0.0, 2).is_none());
        // ln x is undefined at 0.
        assert!(taylor_expand(&var("x").ln_expr(), "x", 0.0, 1).is_none());
        // x^-1 evaluates to +inf at 0, which is not a coefficient.
        assert!(taylor_expand(&var("x").pow_expr(cst(-1.0)), "x", 0.0, 0).is_none());
        // sqrt(x): the value 0 is fine, but the first derivative is infinite.
        let root = var("x").pow_expr(cst(0.5));
        assert!(taylor_expand(&root, "x", 0.0, 0).is_some());
        assert!(taylor_expand(&root, "x", 0.0, 1).is_none());
        // A free variable other than the expansion variable cannot be evaluated.
        let two_vars = var("x").mul_expr(var("y"));
        assert!(taylor_expand(&two_vars, "x", 1.0, 2).is_none());
    }

    #[test]
    fn test_taylor_expand_is_none_when_the_factorial_overflows() {
        let e = var("x").exp_expr();
        assert!(taylor_expand(&e, "x", 0.0, 20).is_some());
        // 171! exceeds f64::MAX, so the coefficient cannot be formed.
        assert!(taylor_expand(&e, "x", 0.0, 171).is_none());
    }
}
