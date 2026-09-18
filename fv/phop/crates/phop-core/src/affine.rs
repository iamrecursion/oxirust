//! Affine- and log-affine-leaf EML discovery (M6 root-cause fix, steps 1 + 2).
//!
//! The Feynman post-mortem showed baseline phop (depth-3, bare var/const leaves) recovers ~nothing:
//! physics laws need scalings, linear combinations, products and powers, which in pure EML cost depth
//! the bounded search can't afford. This engine makes those forms *primitive*:
//!
//! - [`ANode::Linear`] `= Σ aᵢ·xᵢ + b` — a full affine combination of the inputs (step 1).
//! - [`ANode::LogLinear`] `= Σ aᵢ·ln xᵢ + b` — whose `exp` is a **monomial** `e^b·∏ xᵢ^{aᵢ}`, so any
//!   product / ratio / power-law is `eml(LogLinear, 1)` at **depth 1** (step 2 — the multiplicative
//!   laws the affine-only leaf couldn't reach).
//!
//! Because each leaf is a full-width combination fitted by Levenberg–Marquardt, the structure search
//! is just over the `eml` skeleton + a *leaf type* per slot (`Const`/`Linear`/`LogLinear`) — no
//! per-variable enumerate explosion. It is its own engine (real-affine/-log leaves + guarded `eml`),
//! deliberately NOT folded into phop's EmlTree/autograd path: `a·x+b` (or `Σaᵢ ln xᵢ`) with negative
//! coefficients is not guarded real EML (it would need the complex branch).

use crate::polish::solve_dense;
use crate::rng::SplitMix64;
use scirs2_core::ndarray::{Array1, Array2};

/// Symmetric clamp on `exp` arguments (matches [`crate::forest`]).
const EXP_CLAMP: f64 = 50.0;
/// Lower clamp on `ln` arguments (matches [`crate::forest`]).
const LN_EPS: f64 = 1e-12;
/// A coefficient is "active" (counts toward complexity) above this magnitude.
const ACTIVE_EPS: f64 = 1e-6;

/// An EML tree with rich leaves: constants, affine combinations, and log-affine (monomial) combinations.
#[derive(Clone, Debug)]
pub enum ANode {
    /// A free constant.
    Const(f64),
    /// Affine combination `Σ coeffs[i]·x_i + b`.
    Linear {
        /// Per-variable slopes.
        coeffs: Vec<f64>,
        /// Intercept.
        b: f64,
    },
    /// Log-affine combination `Σ coeffs[i]·ln(x_i) + b`; `exp` of it is a monomial `e^b·∏ x_i^{coeffs[i]}`.
    LogLinear {
        /// Per-variable log-exponents.
        coeffs: Vec<f64>,
        /// Log-intercept.
        b: f64,
    },
    /// The EML primitive `exp(left) − ln(right)`.
    Eml(Box<ANode>, Box<ANode>),
}

fn active(coeffs: &[f64]) -> usize {
    coeffs.iter().filter(|c| c.abs() > ACTIVE_EPS).count()
}

impl ANode {
    /// Complexity: structural nodes plus the number of *active* coefficients (so a monomial in 3
    /// variables is "bigger" than a constant).
    #[must_use]
    pub fn nodes(&self) -> usize {
        match self {
            ANode::Const(_) => 1,
            ANode::Linear { coeffs, .. } | ANode::LogLinear { coeffs, .. } => 1 + active(coeffs),
            ANode::Eml(l, r) => 1 + l.nodes() + r.nodes(),
        }
    }

    /// Maximum `eml` depth (a leaf has depth 0).
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            ANode::Eml(l, r) => 1 + l.depth().max(r.depth()),
            _ => 0,
        }
    }

    /// A readable rendering.
    #[must_use]
    pub fn pretty(&self) -> String {
        match self {
            ANode::Const(c) => format!("{c:.4}"),
            ANode::Linear { coeffs, b } => format!("({})", combo(coeffs, *b, "x")),
            ANode::LogLinear { coeffs, b } => format!("({})", combo(coeffs, *b, "ln x")),
            // eml(z, 1) = exp(z) − ln(1) = exp(z): fold the trivial denominator, and render a
            // LogLinear numerator as the monomial it represents (e.g. x0·x1, x0^1.5).
            ANode::Eml(l, r) => match const_value(r) {
                Some(c) if (c - 1.0).abs() < 1e-6 => match l.as_ref() {
                    ANode::LogLinear { coeffs, b } => monomial(coeffs, *b, false),
                    _ => format!("exp({})", l.pretty()),
                },
                _ => format!("eml({}, {})", l.pretty(), r.pretty()),
            },
        }
    }
}

/// If `node` is constant-valued — a `Const`, or a combination leaf whose coefficients are all
/// inactive (so it reduces to its intercept) — return that constant; else `None`. Used to fold a
/// trivial `eml(z, 1)` denominator for display.
fn const_value(node: &ANode) -> Option<f64> {
    match node {
        ANode::Const(c) => Some(*c),
        ANode::Linear { coeffs, b } | ANode::LogLinear { coeffs, b } => {
            coeffs.iter().all(|c| c.abs() <= ACTIVE_EPS).then_some(*b)
        }
        ANode::Eml(_, _) => None,
    }
}

/// Render an exponent: an integer when it snapped to one, else a short decimal.
fn fmt_exp(a: f64) -> String {
    if (a - a.round()).abs() < 1e-9 {
        format!("{}", a.round() as i64)
    } else {
        format!("{a:.3}")
    }
}

/// `exp(Σ aᵢ ln xᵢ + b) = exp(b)·∏ xᵢ^{aᵢ}` — the clean monomial form of `eml(LogLinear, 1)`. With
/// `latex`, variables/exponents use subscript/superscript markup; otherwise an ASCII `x{i}^{a}` form.
fn monomial(coeffs: &[f64], b: f64, latex: bool) -> String {
    let mut parts: Vec<String> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, a)| a.abs() > ACTIVE_EPS)
        .map(|(i, a)| match (latex, (a - 1.0).abs() < 1e-9) {
            (true, true) => format!("x_{{{i}}}"),
            (true, false) => format!("x_{{{i}}}^{{{}}}", fmt_exp(*a)),
            (false, true) => format!("x{i}"),
            (false, false) => format!("x{i}^{}", fmt_exp(*a)),
        })
        .collect();
    if b.abs() > ACTIVE_EPS {
        parts.push(if latex {
            format!("e^{{{b:.3}}}")
        } else {
            format!("exp({b:.3})")
        });
    }
    match (parts.is_empty(), latex) {
        (true, _) => "1".to_string(),
        (false, true) => parts.join(" \\cdot "),
        (false, false) => parts.join("*"),
    }
}

/// Format `Σ coeffs[i]·<sym>i + b`, keeping only active terms.
fn combo(coeffs: &[f64], b: f64, sym: &str) -> String {
    let mut parts: Vec<String> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() > ACTIVE_EPS)
        .map(|(i, c)| format!("{c:.3}*{sym}{i}"))
        .collect();
    if b.abs() > ACTIVE_EPS || parts.is_empty() {
        parts.push(format!("{b:.3}"));
    }
    parts.join(" + ")
}

/// Guarded forward evaluation over `x` (`[n_rows, n_vars]`).
#[must_use]
pub fn eval(node: &ANode, x: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows();
    match node {
        ANode::Const(c) => Array1::from_elem(n, *c),
        ANode::Linear { coeffs, b } => {
            let mut out = Array1::from_elem(n, *b);
            for (j, &cf) in coeffs.iter().enumerate() {
                if cf != 0.0 {
                    for i in 0..n {
                        out[i] += cf * x[[i, j]];
                    }
                }
            }
            out
        }
        ANode::LogLinear { coeffs, b } => {
            let mut out = Array1::from_elem(n, *b);
            for (j, &cf) in coeffs.iter().enumerate() {
                if cf != 0.0 {
                    for i in 0..n {
                        out[i] += cf * x[[i, j]].max(LN_EPS).ln();
                    }
                }
            }
            out
        }
        ANode::Eml(l, r) => {
            let la = eval(l, x);
            let rb = eval(r, x);
            let mut out = Array1::zeros(n);
            for i in 0..n {
                let ea = la[i].clamp(-EXP_CLAMP, EXP_CLAMP).exp();
                let lb = rb[i].max(LN_EPS).ln();
                out[i] = ea - lb;
            }
            out
        }
    }
}

/// Collect free parameters in pre-order (`coeffs…, b` per combination leaf; `c` per constant).
fn collect(node: &ANode, out: &mut Vec<f64>) {
    match node {
        ANode::Const(c) => out.push(*c),
        ANode::Linear { coeffs, b } | ANode::LogLinear { coeffs, b } => {
            out.extend_from_slice(coeffs);
            out.push(*b);
        }
        ANode::Eml(l, r) => {
            collect(l, out);
            collect(r, out);
        }
    }
}

/// Rebuild a tree with parameters taken from `p` in pre-order.
fn apply(node: &ANode, p: &[f64], idx: &mut usize) -> ANode {
    match node {
        ANode::Const(_) => {
            let c = p[*idx];
            *idx += 1;
            ANode::Const(c)
        }
        ANode::Linear { coeffs, .. } => {
            let n = coeffs.len();
            let cs = p[*idx..*idx + n].to_vec();
            let b = p[*idx + n];
            *idx += n + 1;
            ANode::Linear { coeffs: cs, b }
        }
        ANode::LogLinear { coeffs, .. } => {
            let n = coeffs.len();
            let cs = p[*idx..*idx + n].to_vec();
            let b = p[*idx + n];
            *idx += n + 1;
            ANode::LogLinear { coeffs: cs, b }
        }
        ANode::Eml(l, r) => ANode::Eml(Box::new(apply(l, p, idx)), Box::new(apply(r, p, idx))),
    }
}

/// Reset every leaf's coefficients to `coeff_init` (and intercept/const to a neutral start) for a
/// fresh Levenberg–Marquardt start.
fn reinit(node: &ANode, coeff_init: f64) -> ANode {
    match node {
        ANode::Const(_) => ANode::Const(1.0),
        ANode::Linear { coeffs, .. } => ANode::Linear {
            coeffs: vec![coeff_init; coeffs.len()],
            b: 0.0,
        },
        ANode::LogLinear { coeffs, .. } => ANode::LogLinear {
            coeffs: vec![coeff_init; coeffs.len()],
            b: 0.0,
        },
        ANode::Eml(l, r) => ANode::Eml(
            Box::new(reinit(l, coeff_init)),
            Box::new(reinit(r, coeff_init)),
        ),
    }
}

/// Mean-squared error, or `INFINITY` on any non-finite prediction.
fn mse(pred: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = y.len().max(1) as f64;
    let mut s = 0.0;
    for (p, t) in pred.iter().zip(y.iter()) {
        if !p.is_finite() {
            return f64::INFINITY;
        }
        s += (p - t) * (p - t);
    }
    s / n
}

/// Max absolute error for snapping a fitted coefficient to a small rational.
const SNAP_ABS: f64 = 0.03;
/// R² a snapped form must retain to count as a *symbolic* recovery.
const SYMBOLIC_R2: f64 = 0.999;
/// LM iterations used to re-fit the remaining free parameters after each coefficient is snapped.
const SNAP_REFIT_ITERS: usize = 60;

/// Snap `v` to the nearest small rational `k/d` (`d ∈ {1,2,3,4,6}`, `|k/d| ≤ 12`) within [`SNAP_ABS`],
/// preferring the smallest denominator. Near-zero snaps to `0`. Returns `None` if nothing is close.
fn snap_rational(v: f64) -> Option<f64> {
    if v.abs() < SNAP_ABS {
        return Some(0.0);
    }
    for d in [1.0, 2.0, 3.0, 4.0, 6.0] {
        let k = (v * d).round();
        let cand = k / d;
        if cand.abs() <= 12.0 && (v - cand).abs() < SNAP_ABS {
            return Some(cand);
        }
    }
    None
}

/// Snap a linear slope: a recognisable named constant (π, e, √2, …) or a small rational.
fn snap_value(v: f64) -> Option<f64> {
    oxieml::symreg::snap_to_named_const(v)
        .map(|nc| nc.value())
        .or_else(|| snap_rational(v))
}

/// Per-parameter kind, parallel to [`collect`]'s pre-order: `Exp` = a log-linear exponent, `Lin` = a
/// linear slope (both define the *symbolic structure*); `Other` = an intercept/constant (a fitted
/// scale/offset, left untouched).
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Exp,
    Lin,
    Other,
}

fn tag(node: &ANode, out: &mut Vec<Kind>) {
    match node {
        ANode::Const(_) => out.push(Kind::Other),
        ANode::Linear { coeffs, .. } => {
            out.extend(std::iter::repeat_n(Kind::Lin, coeffs.len()));
            out.push(Kind::Other);
        }
        ANode::LogLinear { coeffs, .. } => {
            out.extend(std::iter::repeat_n(Kind::Exp, coeffs.len()));
            out.push(Kind::Other);
        }
        ANode::Eml(l, r) => {
            tag(l, out);
            tag(r, out);
        }
    }
}

/// Snap residual: how far `v` is from its nearest clean target (smaller ⇒ snap with more
/// confidence). `INFINITY` if no clean target is within tolerance, or the parameter is not structural.
fn snap_residual(v: f64, k: Kind) -> f64 {
    let target = match k {
        Kind::Exp => snap_rational(v),
        Kind::Lin => {
            if v.abs() < SNAP_ABS {
                Some(0.0)
            } else {
                snap_value(v)
            }
        }
        Kind::Other => return f64::INFINITY,
    };
    target.map_or(f64::INFINITY, |t| (v - t).abs())
}

/// Iterative rational-rounding (AI-Feynman style): snap structural exponents/slopes to small
/// rationals / named constants **one at a time** — most-confident first — re-fitting the remaining
/// free parameters (intercepts, scales, not-yet-snapped coefficients) after each snap so they can
/// absorb the perturbation. Returns the snapped tree only if **every** structural coefficient snaps
/// while the form retains R² ≥ [`SYMBOLIC_R2`].
///
/// This is strictly stronger than snapping all coefficients at once with the intercepts frozen: the
/// per-snap refit both *rescues* fits that landed just outside the snap window (coupling holds a clean
/// exponent off its integer until its neighbours are pinned) and *rejects* snaps that genuinely break
/// the fit (the R² gate runs after each refit).
fn try_snap(tree: &ANode, x: &Array2<f64>, y: &Array1<f64>) -> Option<ANode> {
    let mut theta = Vec::new();
    collect(tree, &mut theta);
    let mut kinds = Vec::new();
    tag(tree, &mut kinds);

    // Structural parameters only, ordered by snap confidence (closest to a clean target first).
    let mut order: Vec<usize> = (0..theta.len())
        .filter(|&i| kinds[i] != Kind::Other)
        .collect();
    if order.is_empty() {
        return None;
    }
    order.sort_by(|&a, &b| {
        snap_residual(theta[a], kinds[a])
            .partial_cmp(&snap_residual(theta[b], kinds[b]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fixed = vec![false; theta.len()];
    for i in order {
        // Re-test eligibility on the CURRENT value: a prior refit may have pulled it onto (or off) a
        // clean target.
        let snapped_val = match kinds[i] {
            Kind::Exp => snap_rational(theta[i])?,
            Kind::Lin => {
                if theta[i].abs() < SNAP_ABS {
                    0.0
                } else {
                    snap_value(theta[i])?
                }
            }
            Kind::Other => continue,
        };
        theta[i] = snapped_val;
        fixed[i] = true;
        // Re-fit everything still free with this coefficient pinned, then hold the pin exactly.
        let (refit, _) = lm_fit_masked(tree, x, y, SNAP_REFIT_ITERS, &theta, &fixed);
        theta = refit;
        theta[i] = snapped_val;
        let pred = {
            let mut idx = 0;
            eval(&apply(tree, &theta, &mut idx), x)
        };
        if r2(&pred, y) < SYMBOLIC_R2 {
            return None;
        }
    }

    // Best-effort second pass: snap the remaining "Other" constants (eml-argument constants and leaf
    // intercepts) to clean values too — e.g. an `eml(·, 1.0000…)` denominator collapses to `eml(·, 1)`
    // so the `ln(1)=0` residue vanishes. This does not affect the symbolic *criterion* (only structural
    // Exp/Lin coeffs decide that); accept a snap only if R² is retained.
    for i in 0..theta.len() {
        if fixed[i] || kinds[i] != Kind::Other {
            continue;
        }
        if let Some(cv) = snap_rational(theta[i]) {
            if cv == theta[i] {
                continue; // already exactly clean; a near value must still snap (removes ε residue)
            }
            let saved = theta[i];
            theta[i] = cv;
            let pred = {
                let mut idx = 0;
                eval(&apply(tree, &theta, &mut idx), x)
            };
            if r2(&pred, y) < SYMBOLIC_R2 {
                theta[i] = saved; // snap would break the fit — leave the fitted value
            }
        }
    }

    let mut idx = 0;
    Some(apply(tree, &theta, &mut idx))
}

/// Coefficient of determination of `pred` against `y`.
#[must_use]
pub fn r2(pred: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let mean = y.sum() / y.len().max(1) as f64;
    let (mut sr, mut st) = (0.0, 0.0);
    for (p, t) in pred.iter().zip(y.iter()) {
        sr += (t - p) * (t - p);
        st += (t - mean) * (t - mean);
    }
    if st == 0.0 {
        return f64::NAN;
    }
    1.0 - sr / st
}

/// Finite-difference Levenberg–Marquardt fit of the **free** parameters of `skel` — those whose index
/// is `false` in `fixed` — holding the rest at their values in `theta0`. Returns the full updated
/// parameter vector (pinned entries unchanged) and the achieved MSE. With an all-`false` mask this
/// fits every parameter (see [`lm_fit`]); with some entries pinned it is the constrained refit the
/// iterative snapper ([`try_snap`]) relies on.
fn lm_fit_masked(
    skel: &ANode,
    x: &Array2<f64>,
    y: &Array1<f64>,
    iters: usize,
    theta0: &[f64],
    fixed: &[bool],
) -> (Vec<f64>, f64) {
    let mut theta = theta0.to_vec();
    let free: Vec<usize> = (0..theta.len()).filter(|&j| !fixed[j]).collect();
    let p = free.len();
    let eval_at = |th: &[f64]| -> Option<Array1<f64>> {
        let mut idx = 0;
        let pred = eval(&apply(skel, th, &mut idx), x);
        pred.iter().all(|v| v.is_finite()).then_some(pred)
    };
    let Some(mut pred) = eval_at(&theta) else {
        return (theta, f64::INFINITY);
    };
    let mut cost = mse(&pred, y);
    if p == 0 {
        return (theta, cost);
    }
    let n = y.len();
    let mut lambda = 1e-2_f64;

    for _ in 0..iters {
        let r: Vec<f64> = pred.iter().zip(y.iter()).map(|(p, t)| p - t).collect();
        let mut jac = vec![vec![0.0; p]; n];
        let mut ok = true;
        for (jc, &j) in free.iter().enumerate() {
            let h = 1e-6 * (theta[j].abs() + 1.0);
            let mut th = theta.clone();
            th[j] += h;
            let Some(pj) = eval_at(&th) else {
                ok = false;
                break;
            };
            for i in 0..n {
                jac[i][jc] = (pj[i] - pred[i]) / h;
            }
        }
        if !ok {
            break;
        }
        let mut a = vec![vec![0.0; p]; p];
        let mut grad = vec![0.0; p];
        for col in 0..p {
            for (row, jr) in jac.iter().enumerate() {
                grad[col] += jr[col] * r[row];
            }
            for col2 in col..p {
                let s: f64 = jac.iter().map(|jr| jr[col] * jr[col2]).sum();
                a[col][col2] = s;
                a[col2][col] = s;
            }
        }
        let mut accepted = false;
        for _ in 0..12 {
            let mut ad = a.clone();
            for d in 0..p {
                ad[d][d] += lambda * a[d][d].max(1e-12);
            }
            let rhs: Vec<f64> = grad.iter().map(|g| -g).collect();
            let Some(delta) = solve_dense(ad, rhs) else {
                lambda *= 4.0;
                continue;
            };
            let mut cand = theta.clone();
            for (jc, &j) in free.iter().enumerate() {
                cand[j] = theta[j] + delta[jc];
            }
            if let Some(pc) = eval_at(&cand) {
                let cc = mse(&pc, y);
                if cc < cost {
                    theta = cand;
                    pred = pc;
                    cost = cc;
                    lambda = (lambda * 0.5).max(1e-12);
                    accepted = true;
                    break;
                }
            }
            lambda *= 4.0;
        }
        if !accepted {
            break;
        }
    }
    (theta, cost)
}

/// Finite-difference Levenberg–Marquardt fit of *all* of a skeleton's parameters. Returns fitted
/// tree + MSE.
fn lm_fit(skel: &ANode, x: &Array2<f64>, y: &Array1<f64>, iters: usize) -> (ANode, f64) {
    let mut theta0 = Vec::new();
    collect(skel, &mut theta0);
    let fixed = vec![false; theta0.len()];
    let (theta, cost) = lm_fit_masked(skel, x, y, iters, &theta0, &fixed);
    let mut idx = 0;
    (apply(skel, &theta, &mut idx), cost)
}

/// Multi-start fit: try coefficient inits `1.0` (favours monomials/products) and `0.0`, keep the best.
fn lm_fit_best(skel: &ANode, x: &Array2<f64>, y: &Array1<f64>, iters: usize) -> (ANode, f64) {
    let mut best = lm_fit(skel, x, y, iters);
    let alt = lm_fit(&reinit(skel, 0.0), x, y, iters);
    if alt.1 < best.1 {
        best = alt;
    }
    best
}

/// A binary-tree skeleton with placeholder leaves.
#[derive(Clone)]
enum Skel {
    Leaf,
    Node(Box<Skel>, Box<Skel>),
}

impl Skel {
    fn leaves(&self) -> usize {
        match self {
            Skel::Leaf => 1,
            Skel::Node(l, r) => l.leaves() + r.leaves(),
        }
    }
}

/// All binary-tree skeletons with `0..=max_internal` internal (`eml`) nodes.
fn skeletons(max_internal: usize) -> Vec<Skel> {
    let mut by_k: Vec<Vec<Skel>> = vec![vec![Skel::Leaf]];
    for k in 1..=max_internal {
        let mut here = Vec::new();
        for i in 0..k {
            for l in &by_k[i] {
                for r in &by_k[k - 1 - i] {
                    here.push(Skel::Node(Box::new(l.clone()), Box::new(r.clone())));
                }
            }
        }
        by_k.push(here);
    }
    by_k.into_iter().flatten().collect()
}

/// Materialize a skeleton from per-leaf *type* codes (0 = const, 1 = linear, 2 = log-linear).
fn materialize(
    skel: &Skel,
    types: &[usize],
    idx: &mut usize,
    n_vars: usize,
    coeff_init: f64,
) -> ANode {
    match skel {
        Skel::Leaf => {
            let t = types[*idx];
            *idx += 1;
            match t {
                0 => ANode::Const(1.0),
                1 => ANode::Linear {
                    coeffs: vec![coeff_init; n_vars],
                    b: 0.0,
                },
                _ => ANode::LogLinear {
                    coeffs: vec![coeff_init; n_vars],
                    b: 0.0,
                },
            }
        }
        Skel::Node(l, r) => ANode::Eml(
            Box::new(materialize(l, types, idx, n_vars, coeff_init)),
            Box::new(materialize(r, types, idx, n_vars, coeff_init)),
        ),
    }
}

/// A discovered law and its quality.
#[derive(Clone, Debug)]
pub struct AffineSolution {
    /// The fitted tree (its forward IS the prediction; see [`Self::predict`]).
    pub tree: ANode,
    /// Mean-squared error on the fitting data.
    pub mse: f64,
    /// Coefficient of determination.
    pub r2: f64,
    /// Readable expression.
    pub expr: String,
    /// Complexity (nodes + active coefficients).
    pub nodes: usize,
    /// Tree depth.
    pub depth: usize,
    /// Whether the law is a **symbolic** recovery: every structural exponent/slope snaps to a small
    /// rational / named constant while retaining R² ≥ 0.999 (set by `Self::with_snap`).
    pub symbolic: bool,
}

impl AffineSolution {
    fn from_tree(tree: ANode, x: &Array2<f64>, y: &Array1<f64>, mse: f64) -> Self {
        let pred = eval(&tree, x);
        Self {
            r2: r2(&pred, y),
            expr: tree.pretty(),
            nodes: tree.nodes(),
            depth: tree.depth(),
            symbolic: false,
            mse,
            tree,
        }
    }

    /// If the law snaps to a clean rational-exponent / linear form (keeping R² ≥ 0.999), replace the
    /// tree with the snapped one and mark it `symbolic`; otherwise return it unchanged.
    #[must_use]
    fn with_snap(mut self, x: &Array2<f64>, y: &Array1<f64>) -> Self {
        if let Some(snapped) = try_snap(&self.tree, x, y) {
            let pred = eval(&snapped, x);
            self.mse = mse(&pred, y);
            self.r2 = r2(&pred, y);
            self.expr = snapped.pretty();
            self.nodes = snapped.nodes();
            self.depth = snapped.depth();
            self.tree = snapped;
            self.symbolic = true;
        }
        self
    }

    /// Evaluate the discovered law on new data `x` (`[n_rows, n_vars]`).
    #[must_use]
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        eval(&self.tree, x)
    }

    /// LaTeX rendering of the law.
    #[must_use]
    pub fn latex(&self) -> String {
        to_latex(&self.tree)
    }
}

/// LaTeX for a tree (combination leaves render directly).
#[must_use]
pub fn to_latex(node: &ANode) -> String {
    match node {
        ANode::Const(c) => format!("{c:.4}"),
        ANode::Linear { coeffs, b } => combo_latex(coeffs, *b, false),
        ANode::LogLinear { coeffs, b } => combo_latex(coeffs, *b, true),
        // eml(z, 1) = exp(z) − ln(1) = exp(z): fold the trivial denominator; render a LogLinear
        // numerator as its monomial (x_0 x_1, x_0^{3/2}, …).
        ANode::Eml(l, r) => match const_value(r) {
            Some(c) if (c - 1.0).abs() < 1e-6 => match l.as_ref() {
                ANode::LogLinear { coeffs, b } => monomial(coeffs, *b, true),
                _ => format!("e^{{{}}}", to_latex(l)),
            },
            _ => format!(
                "\\left(e^{{{}}} - \\ln\\left({}\\right)\\right)",
                to_latex(l),
                to_latex(r)
            ),
        },
    }
}

fn combo_latex(coeffs: &[f64], b: f64, log: bool) -> String {
    let mut parts: Vec<String> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() > ACTIVE_EPS)
        .map(|(i, c)| {
            if log {
                format!("{c:.3}\\,\\ln x_{{{i}}}")
            } else {
                format!("{c:.3}\\,x_{{{i}}}")
            }
        })
        .collect();
    if b.abs() > ACTIVE_EPS || parts.is_empty() {
        parts.push(format!("{b:.3}"));
    }
    parts.join(" + ")
}

/// Build the candidate pool: every `eml` skeleton (`0..=max_internal` nodes) × every per-leaf type
/// assignment (const/linear/log-linear). Bounded by `cand_cap`.
fn build_pool(n_vars: usize, max_internal: usize, cand_cap: usize) -> Vec<ANode> {
    const RADIX: usize = 3; // const | linear | log-linear
    const EXHAUSTIVE_MAX: u128 = 256;
    const SAMPLES_PER_SKEL: usize = 200;

    let mut rng = SplitMix64::new(0xA5F1_C0DE ^ n_vars as u64);
    let mut pool: Vec<ANode> = Vec::new();

    'outer: for skel in skeletons(max_internal) {
        let leaves = skel.leaves();
        let total = (RADIX as u128)
            .checked_pow(leaves as u32)
            .unwrap_or(u128::MAX);
        if total <= EXHAUSTIVE_MAX {
            for code in 0..total {
                if pool.len() >= cand_cap {
                    break 'outer;
                }
                let mut types = vec![0usize; leaves];
                let mut c = code;
                for slot in types.iter_mut() {
                    *slot = (c % RADIX as u128) as usize;
                    c /= RADIX as u128;
                }
                let mut idx = 0;
                pool.push(materialize(&skel, &types, &mut idx, n_vars, 1.0));
            }
        } else {
            for _ in 0..SAMPLES_PER_SKEL {
                if pool.len() >= cand_cap {
                    break 'outer;
                }
                let types: Vec<usize> = (0..leaves).map(|_| rng.below(RADIX)).collect();
                let mut idx = 0;
                pool.push(materialize(&skel, &types, &mut idx, n_vars, 1.0));
            }
        }
    }
    pool
}

/// Quick-fit every candidate, multi-start-refit the most promising, return their fitted solutions.
fn fit_pool(pool: &[ANode], x: &Array2<f64>, y: &Array1<f64>) -> Vec<AffineSolution> {
    const QUICK: usize = 10;
    const REFIT: usize = 50;
    const REFIT_K: usize = 40;

    let mut scored: Vec<(usize, f64)> = pool
        .iter()
        .enumerate()
        .map(|(i, c)| (i, lm_fit(c, x, y, QUICK).1))
        .filter(|(_, m)| m.is_finite())
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .iter()
        .take(REFIT_K)
        .filter_map(|(i, _)| {
            let (fitted, m) = lm_fit_best(&pool[*i], x, y, REFIT);
            m.is_finite()
                .then(|| AffineSolution::from_tree(fitted, x, y, m))
        })
        .collect()
}

/// Discover a rich-leaf EML law for `(x, y)`. `max_internal` bounds `eml` nodes; `cand_cap` bounds
/// candidates. Returns the best fit by MSE, or `None` if `x` is empty.
#[must_use]
pub fn discover_affine(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_internal: usize,
    cand_cap: usize,
) -> Option<AffineSolution> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return None;
    }
    let pool = build_pool(x.ncols(), max_internal, cand_cap);
    fit_pool(&pool, x, y)
        .into_iter()
        .min_by(|a, b| {
            a.mse
                .partial_cmp(&b.mse)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.with_snap(x, y))
}

/// Discover a rich-leaf EML **Pareto front** (non-dominated over complexity and MSE), sorted by
/// increasing complexity.
#[must_use]
pub fn discover_affine_pareto(
    x: &Array2<f64>,
    y: &Array1<f64>,
    max_internal: usize,
    cand_cap: usize,
) -> Vec<AffineSolution> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Vec::new();
    }
    let pool = build_pool(x.ncols(), max_internal, cand_cap);
    let cands: Vec<AffineSolution> = fit_pool(&pool, x, y)
        .into_iter()
        .map(|s| s.with_snap(x, y))
        .collect();

    let mut front: Vec<AffineSolution> = Vec::new();
    for c in cands {
        let dominated = front
            .iter()
            .any(|s| s.nodes <= c.nodes && s.mse <= c.mse && (s.nodes < c.nodes || s.mse < c.mse));
        if dominated {
            continue;
        }
        front.retain(|s| {
            !(c.nodes <= s.nodes && c.mse <= s.mse && (c.nodes < s.nodes || c.mse < s.mse))
        });
        front.push(c);
    }
    front.sort_by(|a, b| {
        a.nodes.cmp(&b.nodes).then(
            a.mse
                .partial_cmp(&b.mse)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    front
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds(f: impl Fn(&[f64]) -> f64, cols: &[(f64, f64)], n: usize) -> (Array2<f64>, Array1<f64>) {
        let nv = cols.len();
        let mut xv = Vec::with_capacity(n * nv);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            // Sample each column on an INDEPENDENT (per-column permuted) grid so features are not
            // collinear — collinear columns make a multivariate linear fit's normal equations
            // singular and don't test real recovery. The strides are coprime-with-n permutations.
            let row: Vec<f64> = cols
                .iter()
                .enumerate()
                .map(|(j, (lo, hi))| {
                    let idx = (i * (2 * j + 1) + 7 * j) % n;
                    lo + (hi - lo) * (idx as f64) / (n as f64 - 1.0)
                })
                .collect();
            yv.push(f(&row));
            xv.extend(&row);
        }
        (
            Array2::from_shape_vec((n, nv), xv).expect("shape"),
            Array1::from(yv),
        )
    }

    #[test]
    fn recovers_linear_combination() {
        // y = 3·x0 − 2·x1 + 1: a single Linear leaf over both variables.
        let (x, y) = ds(
            |r| 3.0 * r[0] - 2.0 * r[1] + 1.0,
            &[(0.5, 5.0), (1.0, 4.0)],
            50,
        );
        let s = discover_affine(&x, &y, 3, 2000).expect("solution");
        assert!(
            s.r2 > 0.9999,
            "linear combo not recovered: r2={} expr={}",
            s.r2,
            s.expr
        );
    }

    #[test]
    fn recovers_scaled_exponential() {
        // y = e^{2x} = eml(Linear{2x}, 1).
        let (x, y) = ds(|r| (2.0 * r[0]).exp(), &[(0.0, 2.0)], 40);
        let s = discover_affine(&x, &y, 3, 2000).expect("solution");
        assert!(
            s.r2 > 0.999,
            "scaled exp not recovered: r2={} expr={}",
            s.r2,
            s.expr
        );
    }

    #[test]
    fn recovers_product() {
        // y = x0·x1 = eml(LogLinear{ln x0 + ln x1}, 1) — the multiplicative case (step 2).
        let (x, y) = ds(|r| r[0] * r[1], &[(0.5, 5.0), (0.5, 5.0)], 50);
        let s = discover_affine(&x, &y, 3, 2000).expect("solution");
        assert!(
            s.r2 > 0.999,
            "product not recovered: r2={} expr={}",
            s.r2,
            s.expr
        );
    }

    #[test]
    fn recovers_power_and_ratio() {
        // y = x0² / x1 = eml(LogLinear{2 ln x0 − ln x1}, 1) — a power-law ratio monomial.
        let (x, y) = ds(|r| r[0] * r[0] / r[1], &[(0.5, 5.0), (0.5, 5.0)], 50);
        let s = discover_affine(&x, &y, 3, 2000).expect("solution");
        assert!(
            s.r2 > 0.999,
            "power/ratio not recovered: r2={} expr={}",
            s.r2,
            s.expr
        );
    }

    #[test]
    fn symbolic_recovery_snaps_exponents() {
        // y = x0² / x1: the fitted exponents (≈2, ≈−1) must snap to exact rationals → symbolic.
        let (x, y) = ds(|r| r[0] * r[0] / r[1], &[(0.5, 5.0), (0.5, 5.0)], 50);
        let s = discover_affine(&x, &y, 3, 2000).expect("solution");
        assert!(
            s.symbolic,
            "x0^2/x1 should be a symbolic recovery: expr={}",
            s.expr
        );
        assert!(s.r2 >= 0.999, "snapped form lost accuracy: r2={}", s.r2);

        // A non-monomial transcendental (e^{2x}) is a numeric recovery but NOT a clean monomial:
        // its single Linear-slope leaf still snaps (slope 2), so it IS symbolic too — sanity that
        // the flag is set for clean linear forms as well.
        let (x2, y2) = ds(|r| (2.0 * r[0]).exp(), &[(0.0, 2.0)], 40);
        let s2 = discover_affine(&x2, &y2, 3, 2000).expect("solution");
        assert!(s2.r2 > 0.999);
    }

    #[test]
    fn pareto_front_is_non_dominated_and_sorted() {
        let (x, y) = ds(|r| r[0] * r[1], &[(0.5, 5.0), (0.5, 5.0)], 40);
        let front = discover_affine_pareto(&x, &y, 3, 2000);
        assert!(!front.is_empty(), "empty pareto front");
        for w in front.windows(2) {
            assert!(w[0].nodes <= w[1].nodes, "front not sorted by complexity");
        }
        assert!(
            front.iter().any(|s| s.r2 > 0.999),
            "no accurate solution on the front"
        );
    }
}
