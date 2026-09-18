//! Proof-carrying tier via **OxiLean** — the cool-japan pure-Rust Lean theorem-prover kernel.
//!
//! Where [`crate::analyze`] (interval certificates) and [`crate::verify`] (SMT, feature `smt`)
//! establish *numeric* and *logical* facts, this module produces a **machine-checked** artifact: a
//! property is stated as a Lean proposition and a proof term is handed to the `oxilean-kernel`
//! type-checker, the same trusted kernel that underpins Lean. If the kernel accepts the
//! `Declaration::Theorem`, the proof is verified.
//!
//! ## What is (and isn't) proven — read this
//!
//! OxiLean 0.1.2's default kernel environment has **no real-number theory and no computation**
//! (Nat arithmetic builtins are opaque axioms; there is no `Real.exp`/`Real.log`/`rpow` anywhere in
//! the ecosystem yet). So we *cannot* yet prove a real-analytic fact such as `a^{3/2} = exp(1.5·ln a)`
//! over ℝ. What we **can** do — and do here — is machine-check that phop's symbolic **lowering step**
//! is a valid *equational rewrite* under a small, explicit, postulated EML theory:
//!
//! ```text
//!   carrier  R : Type
//!   ops      exp ln eml sub : R → …,   one zero : R
//!   axioms   eml_def : ∀ x y, eml x y = sub (exp x) (ln y)
//!            ln_one  : ln one = zero
//!            sub_zero: ∀ a,   sub a zero = a
//!            eq_trans, eq_congr             (standard equality lemmas, postulated)
//!   theorem  ∀ x, eml x one = exp x        ← kernel-checked rewrite chain
//! ```
//!
//! The kernel verifies the **rewrite chain composes with correct types** (it rejects a wrong chain,
//! and rejects a proof whose type doesn't match the stated goal — see the negative control). This is
//! a genuine machine-checked proof, but it is **relative to the postulated axioms**: it certifies the
//! lowering is logically valid, *not* that `exp`/`ln` are correctly realised over ℝ. It is therefore
//! **strictly weaker** than phop's unconditional interval-arithmetic [`crate::verify::Verdict::Proven`].
//! Enable with the `lean` feature.

use oxilean_kernel::{
    check_declaration, init_builtin_env, BinderInfo, Declaration, Environment, Expr, Level, Name,
};

/// A named-variable term; lowered to de Bruijn [`Expr`] by [`build`] so we never hand-write indices.
#[derive(Clone)]
enum T {
    /// A locally bound variable, referenced by name.
    Var(String),
    /// A global constant with explicit universe-level arguments (empty for monomorphic constants).
    Con(String, Vec<Level>),
    /// A sort (`Sort l`).
    Sort(Level),
    /// Application.
    App(Box<T>, Box<T>),
    /// Dependent function type `(name : dom) → body`.
    Pi(String, Box<T>, Box<T>),
    /// Lambda `fun (name : dom) => body`.
    Lam(String, Box<T>, Box<T>),
}

fn var(s: &str) -> T {
    T::Var(s.into())
}
fn c(s: &str) -> T {
    T::Con(s.into(), vec![])
}
fn app(f: T, a: T) -> T {
    T::App(Box::new(f), Box::new(a))
}
fn app2(f: T, a: T, b: T) -> T {
    app(app(f, a), b)
}
fn appn(f: T, args: Vec<T>) -> T {
    args.into_iter().fold(f, app)
}
fn pi(n: &str, dom: T, body: T) -> T {
    T::Pi(n.into(), Box::new(dom), Box::new(body))
}
/// Non-dependent function type; the binder is still tracked so de Bruijn indices stay correct.
fn arrow(dom: T, body: T) -> T {
    pi("_", dom, body)
}
fn lam(n: &str, dom: T, body: T) -> T {
    T::Lam(n.into(), Box::new(dom), Box::new(body))
}

/// Universe level `1` (so `R : Type 0 = Sort 1`, and `Eq.{1}` works over `R`).
fn u1() -> Level {
    Level::succ(Level::zero())
}
/// `Eq.{1} carrier a b`.
fn eq_of(carrier: T, a: T, b: T) -> T {
    appn(T::Con("Eq".into(), vec![u1()]), vec![carrier, a, b])
}
/// `Eq.{1} R a b` (the EML carrier).
fn eq(a: T, b: T) -> T {
    eq_of(c("R"), a, b)
}
/// `Eq.refl.{1} carrier a`.
fn refl_of(carrier: T, a: T) -> T {
    appn(T::Con("Eq.refl".into(), vec![u1()]), vec![carrier, a])
}

/// Lower a named term to a de Bruijn [`Expr`]. Returns `Err` only on an unbound name (a build bug).
fn to_expr(t: &T, scope: &mut Vec<String>) -> Result<Expr, String> {
    Ok(match t {
        T::Var(name) => {
            let pos = scope
                .iter()
                .rposition(|s| s == name)
                .ok_or_else(|| format!("unbound variable `{name}`"))?;
            Expr::BVar((scope.len() - 1 - pos) as u32)
        }
        T::Con(name, levels) => Expr::Const(Name::str(name), levels.clone()),
        T::Sort(l) => Expr::Sort(l.clone()),
        T::App(f, a) => Expr::App(Box::new(to_expr(f, scope)?), Box::new(to_expr(a, scope)?)),
        T::Pi(n, d, b) => {
            let de = to_expr(d, scope)?;
            scope.push(n.clone());
            let be = to_expr(b, scope);
            scope.pop();
            Expr::Pi(
                BinderInfo::Default,
                Name::str(n),
                Box::new(de),
                Box::new(be?),
            )
        }
        T::Lam(n, d, b) => {
            let de = to_expr(d, scope)?;
            scope.push(n.clone());
            let be = to_expr(b, scope);
            scope.pop();
            Expr::Lam(
                BinderInfo::Default,
                Name::str(n),
                Box::new(de),
                Box::new(be?),
            )
        }
    })
}
fn build(t: &T) -> Result<Expr, String> {
    to_expr(t, &mut Vec::new())
}

/// Add a postulated axiom `name : ty`, kernel-checking that `ty` is a valid type.
fn axiom(env: &mut Environment, name: &str, ty: &T) -> Result<(), String> {
    check_declaration(
        env,
        Declaration::Axiom {
            name: Name::str(name),
            univ_params: vec![],
            ty: build(ty)?,
        },
    )
    .map_err(|err| format!("axiom `{name}` rejected: {err:?}"))
}

/// Try to add a theorem `name : goal := proof`; `Ok(true)` iff the kernel accepts the proof.
fn try_theorem(env: &mut Environment, name: &str, goal: &T, proof: &T) -> Result<bool, String> {
    let decl = Declaration::Theorem {
        name: Name::str(name),
        univ_params: vec![],
        ty: build(goal)?,
        val: build(proof)?,
    };
    Ok(check_declaration(env, decl).is_ok())
}

/// Outcome of an OxiLean proof check.
#[derive(Debug, Clone)]
pub struct LeanProof {
    /// Human-readable statement that was checked.
    pub theorem: String,
    /// The postulated theory the proof is relative to.
    pub axioms: Vec<String>,
    /// Whether the kernel accepted the proof of `theorem`.
    pub accepted: bool,
    /// Whether a deliberately-wrong goal (same proof term) was correctly **rejected** — evidence the
    /// kernel is checking, not rubber-stamping.
    pub negative_control_rejected: bool,
    /// The soundness caveat: this proof is relative to the postulated axioms.
    pub note: String,
}

/// **Tier 1 — the kernel is a genuine checker.** Spins up the builtin environment and confirms the
/// kernel *accepts* a true proof (`Eq.refl : Eq Nat 0 0`) and *rejects* a false one
/// (`Eq.refl : Eq Nat 0 1`). Returns `Ok(true)` iff both hold (i.e. the prover wiring is sound).
pub fn kernel_self_check() -> Result<bool, String> {
    let mut env = Environment::new();
    init_builtin_env(&mut env).map_err(|e| format!("init_builtin_env: {e}"))?;

    let nat = c("Nat");
    let zero = c("Nat.zero");
    let one = app(c("Nat.succ"), zero.clone());

    let good = try_theorem(
        &mut env,
        "phop_zero_eq_zero",
        &eq_of(nat.clone(), zero.clone(), zero.clone()),
        &refl_of(nat.clone(), zero.clone()),
    )?;
    let bad = try_theorem(
        &mut env,
        "phop_zero_eq_one",
        &eq_of(nat.clone(), zero.clone(), one),
        &refl_of(nat, zero),
    )?;
    Ok(good && !bad)
}

/// **Tier 2 — proof-carrying EML lowering.** Postulates the minimal EML theory and has the OxiLean
/// kernel machine-check the rewrite chain for `∀ x, eml x one = exp x` (phop's lowering of the
/// `eml(x, 1)` tree, since `ln 1 = 0`). Also runs a negative control: the *same* proof term against
/// the wrong goal `eml x one = ln x` must be rejected.
///
/// See the module docs: the result is **machine-checked relative to the postulated axioms**, weaker
/// than the unconditional interval-arithmetic guarantee.
pub fn prove_eml_one_lowering() -> Result<LeanProof, String> {
    let mut env = Environment::new();
    init_builtin_env(&mut env).map_err(|e| format!("init_builtin_env: {e}"))?;

    let r = || c("R");
    // carrier + operations
    axiom(&mut env, "R", &T::Sort(u1()))?;
    axiom(&mut env, "one", &r())?;
    axiom(&mut env, "zero", &r())?;
    axiom(&mut env, "exp", &arrow(r(), r()))?;
    axiom(&mut env, "ln", &arrow(r(), r()))?;
    axiom(&mut env, "eml", &arrow(r(), arrow(r(), r())))?;
    axiom(&mut env, "sub", &arrow(r(), arrow(r(), r())))?;
    // EML domain axioms
    axiom(
        &mut env,
        "eml_def",
        &pi(
            "x",
            r(),
            pi(
                "y",
                r(),
                eq(
                    app2(c("eml"), var("x"), var("y")),
                    app2(c("sub"), app(c("exp"), var("x")), app(c("ln"), var("y"))),
                ),
            ),
        ),
    )?;
    axiom(&mut env, "ln_one", &eq(app(c("ln"), c("one")), c("zero")))?;
    axiom(
        &mut env,
        "sub_zero",
        &pi("a", r(), eq(app2(c("sub"), var("a"), c("zero")), var("a"))),
    )?;
    // standard equality lemmas (postulated)
    axiom(
        &mut env,
        "eq_trans",
        &pi(
            "a",
            r(),
            pi(
                "b",
                r(),
                pi(
                    "d",
                    r(),
                    arrow(
                        eq(var("a"), var("b")),
                        arrow(eq(var("b"), var("d")), eq(var("a"), var("d"))),
                    ),
                ),
            ),
        ),
    )?;
    axiom(
        &mut env,
        "eq_congr",
        &pi(
            "f",
            arrow(r(), r()),
            pi(
                "a",
                r(),
                pi(
                    "b",
                    r(),
                    arrow(
                        eq(var("a"), var("b")),
                        eq(app(var("f"), var("a")), app(var("f"), var("b"))),
                    ),
                ),
            ),
        ),
    )?;

    // term-level R values (with `x` in scope)
    let a_eml = || app2(c("eml"), var("x"), c("one")); // eml x one
    let b_sub = || app2(c("sub"), app(c("exp"), var("x")), app(c("ln"), c("one"))); // sub (exp x)(ln one)
    let d_sub = || app2(c("sub"), app(c("exp"), var("x")), c("zero")); // sub (exp x) zero
    let c_exp = || app(c("exp"), var("x")); // exp x

    // rewrite chain:  eml x one =[eml_def] sub(exp x)(ln one) =[congr·ln_one] sub(exp x) zero =[sub_zero] exp x
    let step1 = app2(c("eml_def"), var("x"), c("one"));
    let congr_f = lam("t", r(), app2(c("sub"), app(c("exp"), var("x")), var("t")));
    let step2 = appn(
        c("eq_congr"),
        vec![congr_f, app(c("ln"), c("one")), c("zero"), c("ln_one")],
    );
    let step3 = app(c("sub_zero"), app(c("exp"), var("x")));
    let inner = appn(c("eq_trans"), vec![b_sub(), d_sub(), c_exp(), step2, step3]);
    let outer = appn(c("eq_trans"), vec![a_eml(), b_sub(), c_exp(), step1, inner]);

    let goal = pi("x", r(), eq(a_eml(), c_exp()));
    let proof = lam("x", r(), outer);
    let accepted = try_theorem(&mut env, "phop_eml_one_eq_exp", &goal, &proof)?;

    // negative control: the same proof must NOT certify a different goal.
    let wrong_goal = pi("x", r(), eq(a_eml(), app(c("ln"), var("x"))));
    let wrong_proof = lam("x", r(), {
        // rebuild the chain (the previous `outer` was moved into `proof`)
        let step1 = app2(c("eml_def"), var("x"), c("one"));
        let congr_f = lam("t", r(), app2(c("sub"), app(c("exp"), var("x")), var("t")));
        let step2 = appn(
            c("eq_congr"),
            vec![congr_f, app(c("ln"), c("one")), c("zero"), c("ln_one")],
        );
        let step3 = app(c("sub_zero"), app(c("exp"), var("x")));
        let inner = appn(c("eq_trans"), vec![b_sub(), d_sub(), c_exp(), step2, step3]);
        appn(c("eq_trans"), vec![a_eml(), b_sub(), c_exp(), step1, inner])
    });
    let negative_control_rejected = !try_theorem(
        &mut env,
        "phop_eml_one_eq_ln_BOGUS",
        &wrong_goal,
        &wrong_proof,
    )?;

    Ok(LeanProof {
        theorem: "∀ x : R, eml x one = exp x".to_string(),
        axioms: vec![
            "R : Type".into(),
            "one, zero : R".into(),
            "exp, ln : R → R".into(),
            "eml, sub : R → R → R".into(),
            "eml_def : ∀ x y, eml x y = sub (exp x) (ln y)".into(),
            "ln_one : ln one = zero".into(),
            "sub_zero : ∀ a, sub a zero = a".into(),
            "eq_trans, eq_congr (standard equality lemmas)".into(),
        ],
        accepted,
        negative_control_rejected,
        note: "Machine-checked by the OxiLean kernel RELATIVE TO the postulated EML axioms above: it \
               certifies the lowering rewrite is logically valid, not that exp/ln are realised over ℝ. \
               Strictly weaker than phop's unconditional interval-arithmetic `Verdict::Proven`."
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_accepts_truth_rejects_falsehood() {
        assert_eq!(
            kernel_self_check(),
            Ok(true),
            "kernel must accept Eq Nat 0 0 and reject Eq Nat 0 1"
        );
    }

    #[test]
    fn eml_lowering_is_machine_checked() {
        let proof = prove_eml_one_lowering().expect("PoC must build");
        assert!(
            proof.accepted,
            "kernel must accept the eml(x,1)=exp(x) rewrite under the EML axioms"
        );
        assert!(
            proof.negative_control_rejected,
            "kernel must reject the same proof against the wrong goal (eml x 1 = ln x)"
        );
        assert!(!proof.axioms.is_empty());
    }
}
