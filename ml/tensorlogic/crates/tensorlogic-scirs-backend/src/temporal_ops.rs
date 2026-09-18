//! Temporal logic operators: Next (X), Until (U), WeakUntil (W), Release (R), StrongRelease (M).
//!
//! Provides pure-Rust tensor primitives for compiling the core temporal-logic
//! operators that require sequential (scan) operations not expressible in a single
//! einsum:
//!
//! * **Next** – shifts a tensor by one step along the time axis.
//! * **Until** – computes the backward-scan that satisfies `a U b` semantics.
//! * **WeakUntil** – same recurrence as Until but boundary = 1.0 (top).
//! * **Release** – dual of Until; OUTER=AND, INNER=OR, boundary = 1.0 (top).
//! * **StrongRelease** – dual of WeakUntil; OUTER=AND, INNER=OR, boundary = 0.0 (bottom).
//!
//! All four binary operators share the unified recurrence:
//! ```text
//! u[k] = OUTER(b[k], INNER(a[k], u[k+1]))   for k = T-2..0
//! u[T-1] = OUTER(b[T-1], INNER(a[T-1], boundary_val))
//! ```
//!
//! | Operator         | OUTER | INNER | boundary_val |
//! |------------------|-------|-------|--------------|
//! | Until (U)        | OR    | AND   | 0.0 (bottom) |
//! | WeakUntil (W)    | OR    | AND   | 1.0 (top)    |
//! | Release (R)      | AND   | OR    | 1.0 (top)    |
//! | StrongRelease(M) | AND   | OR    | 0.0 (bottom) |
//!
//! MaxMin semantics: OR=`max`, AND=`min`.
//! ProbSumProduct semantics: OR(x,y)=`x+y-x*y`, AND(x,y)=`x*y`.
//!
//! Two semantics for all operators are supported:
//! * [`UntilSemantics::MaxMin`]  – classical crisp/max-min lattice; non-differentiable.
//! * [`UntilSemantics::ProbSumProduct`] – smooth probabilistic semantics; fully differentiable.
//!
//! The module also exposes VJP helpers (`shift_prev`, `temporal_binary_scan_vjp`) required
//! for backpropagation through these operations.

use scirs2_core::ndarray::{ArrayD, ArrayViewD, Axis, Zip};

use crate::error::TlBackendError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Semantics to use when evaluating temporal binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UntilSemantics {
    /// Classical Gödel / max-min semantics.
    ///
    /// `u[k] = max(b[k], min(a[k], u[k+1]))`
    ///
    /// Not differentiable at the boundary points where the max or min is tied.
    MaxMin,

    /// Smooth probabilistic sum-product semantics.
    ///
    /// `u[k] = b[k] + a[k]*u[k+1] - b[k]*a[k]*u[k+1]`
    ///
    /// Fully differentiable; gradients are well-defined everywhere.
    ProbSumProduct,
}

impl UntilSemantics {
    /// Parse from the tag string embedded in the op name.
    ///
    /// `"max"` → `MaxMin`, `"prod"` → `ProbSumProduct`.
    pub fn from_tag(tag: &str) -> Result<Self, TlBackendError> {
        match tag {
            "max" => Ok(UntilSemantics::MaxMin),
            "prod" => Ok(UntilSemantics::ProbSumProduct),
            other => Err(TlBackendError::invalid_operation(format!(
                "Unknown UntilSemantics tag '{}'; expected 'max' or 'prod'",
                other
            ))),
        }
    }
}

/// Which binary temporal operator is being computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBinaryForm {
    Until,
    WeakUntil,
    Release,
    StrongRelease,
}

impl TemporalBinaryForm {
    /// Parse from the op-string word (e.g. `"until"`, `"weakuntil"`, etc.).
    pub fn from_op_word(word: &str) -> Result<Self, TlBackendError> {
        match word {
            "until" => Ok(Self::Until),
            "weakuntil" => Ok(Self::WeakUntil),
            "release" => Ok(Self::Release),
            "strongrelease" => Ok(Self::StrongRelease),
            other => Err(TlBackendError::invalid_operation(format!(
                "Unknown temporal binary form '{}'; expected until/weakuntil/release/strongrelease",
                other
            ))),
        }
    }

    /// True if the boundary value (past the last time step) is 1.0 (top).
    fn boundary_is_top(self) -> bool {
        matches!(self, Self::WeakUntil | Self::Release)
    }

    /// True if OUTER is OR-like (Until, WeakUntil); false if AND-like (Release, StrongRelease).
    fn outer_is_or(self) -> bool {
        matches!(self, Self::Until | Self::WeakUntil)
    }
}

// ---------------------------------------------------------------------------
// Internal enum
// ---------------------------------------------------------------------------

/// Parsed temporal operation descriptor.
pub(crate) enum TemporalOp {
    /// `temporal_next:<axis>` – shift forward by one step along the given axis.
    Next { axis: usize },
    /// `temporal_<word>:<tag>:<axis>` – backward scan for any binary temporal operator.
    Binary {
        axis: usize,
        sem: UntilSemantics,
        form: TemporalBinaryForm,
    },
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse the `<tag>:<axis>` suffix common to all binary temporal ops.
fn parse_tag_axis(
    remainder: &str,
    op_prefix: &str,
) -> Result<(UntilSemantics, usize), TlBackendError> {
    let mut parts = remainder.splitn(2, ':');
    let tag = parts.next().ok_or_else(|| {
        TlBackendError::invalid_operation(format!("{}: missing tag in '{}'", op_prefix, remainder))
    })?;
    let axis_str = parts.next().ok_or_else(|| {
        TlBackendError::invalid_operation(format!("{}: missing axis in '{}'", op_prefix, remainder))
    })?;
    let sem = UntilSemantics::from_tag(tag)?;
    let axis = axis_str.parse::<usize>().map_err(|_| {
        TlBackendError::invalid_operation(format!(
            "{}: invalid axis '{}' (must be a non-negative integer)",
            op_prefix, axis_str
        ))
    })?;
    Ok((sem, axis))
}

/// Try to parse `op` as a temporal operation.
///
/// Returns:
/// - `None` if `op` does not start with `"temporal_"`.
/// - `Some(Ok(top))` on a successful parse.
/// - `Some(Err(e))` if the prefix matches but the remainder is malformed.
pub(crate) fn parse_temporal_op(op: &str) -> Option<Result<TemporalOp, TlBackendError>> {
    if !op.starts_with("temporal_") {
        return None;
    }

    let rest = &op["temporal_".len()..];

    if let Some(axis_str) = rest.strip_prefix("next:") {
        let axis = match axis_str.parse::<usize>() {
            Ok(a) => a,
            Err(_) => {
                return Some(Err(TlBackendError::invalid_operation(format!(
                    "temporal_next: invalid axis '{}' (must be a non-negative integer)",
                    axis_str
                ))))
            }
        };
        return Some(Ok(TemporalOp::Next { axis }));
    }

    // Binary operators: until, weakuntil, release, strongrelease
    let binary_words = [
        ("until:", TemporalBinaryForm::Until),
        ("weakuntil:", TemporalBinaryForm::WeakUntil),
        ("release:", TemporalBinaryForm::Release),
        ("strongrelease:", TemporalBinaryForm::StrongRelease),
    ];

    for (prefix, form) in &binary_words {
        if let Some(remainder) = rest.strip_prefix(prefix) {
            let op_name = format!("temporal_{}", prefix.trim_end_matches(':'));
            return Some(parse_tag_axis(remainder, &op_name).map(|(sem, axis)| {
                TemporalOp::Binary {
                    axis,
                    sem,
                    form: *form,
                }
            }));
        }
    }

    Some(Err(TlBackendError::invalid_operation(format!(
        "Unknown temporal operation '{}'; expected 'temporal_next:<axis>' or \
         'temporal_<until|weakuntil|release|strongrelease>:<tag>:<axis>'",
        op
    ))))
}

// ---------------------------------------------------------------------------
// Forward primitives
// ---------------------------------------------------------------------------

/// Shift a tensor forward in time along `axis` by one step.
///
/// `out[k] = x[k+1]` for `k = 0 .. T-2`; the last slice along `axis` is zero.
///
/// Special cases:
/// * `T = 0` or `T = 1` → return an all-zero tensor with the same shape.
pub fn shift_next(x: &ArrayViewD<f64>, axis: usize) -> ArrayD<f64> {
    let t = x.len_of(Axis(axis));
    let mut out = ArrayD::zeros(x.raw_dim());

    if t <= 1 {
        return out;
    }

    // Copy x[k+1] → out[k] for k = 0..T-2
    for k in 0..(t - 1) {
        let src = x.index_axis(Axis(axis), k + 1);
        let mut dst = out.index_axis_mut(Axis(axis), k);
        dst.assign(&src);
    }
    // Last slice stays zero (already initialised).
    out
}

/// Evaluate any binary temporal operator (`a OP b`) via the unified backward scan.
///
/// Recurrence (right-to-left, last step to first):
/// ```text
/// u[T-1] = OUTER(b[T-1], INNER(a[T-1], boundary_val))
/// u[k]   = OUTER(b[k],   INNER(a[k],   u[k+1]))         for k = T-2..0
/// ```
///
/// The OUTER/INNER closures and boundary value are determined by `form` and `sem`
/// following the table in the module documentation.
pub fn temporal_binary_scan(
    a: &ArrayViewD<f64>,
    b: &ArrayViewD<f64>,
    axis: usize,
    form: TemporalBinaryForm,
    sem: UntilSemantics,
) -> ArrayD<f64> {
    let t = a.len_of(Axis(axis));
    let mut out = ArrayD::zeros(a.raw_dim());

    if t == 0 {
        return out;
    }

    let boundary_val = if form.boundary_is_top() {
        1.0_f64
    } else {
        0.0_f64
    };
    let outer_or = form.outer_is_or();

    // Compute a single output element given b_v, inner_val, and the current semantics/form.
    // outer_or=true  → OUTER=OR-like: Max or prob-OR
    // outer_or=false → OUTER=AND-like: Min or prob-AND
    let apply_outer = |b_v: f64, inner_val: f64| -> f64 {
        match (outer_or, sem) {
            (true, UntilSemantics::MaxMin) => b_v.max(inner_val),
            (true, UntilSemantics::ProbSumProduct) => b_v + inner_val - b_v * inner_val,
            (false, UntilSemantics::MaxMin) => b_v.min(inner_val),
            (false, UntilSemantics::ProbSumProduct) => b_v * inner_val,
        }
    };

    // INNER is the dual of OUTER:
    // outer_or=true  → INNER=AND-like: Min or prob-AND
    // outer_or=false → INNER=OR-like:  Max or prob-OR
    let apply_inner = |a_v: f64, u_next: f64| -> f64 {
        match (outer_or, sem) {
            (true, UntilSemantics::MaxMin) => a_v.min(u_next),
            (true, UntilSemantics::ProbSumProduct) => a_v * u_next,
            (false, UntilSemantics::MaxMin) => a_v.max(u_next),
            (false, UntilSemantics::ProbSumProduct) => a_v + u_next - a_v * u_next,
        }
    };

    // Base case: u[T-1] = OUTER(b[T-1], INNER(a[T-1], boundary_val))
    {
        let a_last = a.index_axis(Axis(axis), t - 1);
        let b_last = b.index_axis(Axis(axis), t - 1);
        let mut out_last = out.index_axis_mut(Axis(axis), t - 1);
        Zip::from(&mut out_last)
            .and(&b_last)
            .and(&a_last)
            .for_each(|o, &b_v, &a_v| {
                let inner_val = apply_inner(a_v, boundary_val);
                *o = apply_outer(b_v, inner_val);
            });
    }

    // Backward scan: k = T-2 .. 0
    for k in (0..t.saturating_sub(1)).rev() {
        let a_k = a.index_axis(Axis(axis), k);
        let b_k = b.index_axis(Axis(axis), k);
        let u_next = out.index_axis(Axis(axis), k + 1).to_owned();

        let new_slice = Zip::from(&b_k)
            .and(&a_k)
            .and(&u_next)
            .map_collect(|&b_v, &a_v, &u_v| {
                let inner_val = apply_inner(a_v, u_v);
                apply_outer(b_v, inner_val)
            });

        let mut out_k = out.index_axis_mut(Axis(axis), k);
        out_k.assign(&new_slice);
    }

    out
}

/// Vector-Jacobian product (backward pass) for [`temporal_binary_scan`].
///
/// Returns `(grad_a, grad_b)` given the upstream gradient `grad_out`.
///
/// Re-computes the forward pass, then performs a forward-in-time adjoint
/// accumulation sweep (same structure as the existing `until_scan_vjp`).
///
/// For each step `k`, let `i_k = INNER(a[k], u_next_k)` and `u_k = OUTER(b[k], i_k)`.
///
/// **ProbSumProduct — outer_is_or (U/W)**:
/// - `du/di = 1 - b`, `du/db = 1 - i`, `di/da = u_next`, `di/du_next = a`
///
/// **ProbSumProduct — !outer_is_or (R/M)** (OUTER=prob-AND, INNER=prob-OR):
/// - `du/di = b`,     `du/db = i`,     `di/da = 1 - u_next`, `di/du_next = 1 - a`
///
/// **MaxMin — outer_is_or (U/W)**:
/// - `du/db = [b >= i]`, `du/di = [b < i]` (subgradient indicator)
/// - `di/da = [a <= u_next]`, `di/du_next = [u_next < a]`
///
/// **MaxMin — !outer_is_or (R/M)**:
/// - `du/db = [b <= i]`, `du/di = [b > i]`
/// - `di/da = [a >= u_next]`, `di/du_next = [u_next > a]`
pub fn temporal_binary_scan_vjp(
    a: &ArrayViewD<f64>,
    b: &ArrayViewD<f64>,
    grad_out: &ArrayViewD<f64>,
    axis: usize,
    form: TemporalBinaryForm,
    sem: UntilSemantics,
) -> (ArrayD<f64>, ArrayD<f64>) {
    let t = a.len_of(Axis(axis));
    let mut grad_a = ArrayD::zeros(a.raw_dim());
    let mut grad_b = ArrayD::zeros(b.raw_dim());

    if t == 0 {
        return (grad_a, grad_b);
    }

    // Re-compute forward values.
    let u = temporal_binary_scan(a, b, axis, form, sem);

    let boundary_val = if form.boundary_is_top() {
        1.0_f64
    } else {
        0.0_f64
    };
    let outer_or = form.outer_is_or();

    // Adjoint storage seeded from grad_out.
    let mut s = ArrayD::zeros(a.raw_dim());
    s.assign(grad_out);

    // Helper: compute local partial derivatives for one scalar triple (b_v, a_v, u_next_v).
    // Returns (d_outer_d_b, d_outer_d_inner, d_inner_d_a, d_inner_d_u_next).
    let local_partials = |b_v: f64, a_v: f64, u_next_v: f64| -> (f64, f64, f64, f64) {
        match (outer_or, sem) {
            // OUTER = prob-OR: f(b,i) = b + i - b*i;  df/db = 1-i, df/di = 1-b
            // INNER = prob-AND: g(a,u) = a*u;           dg/da = u,   dg/du = a
            (true, UntilSemantics::ProbSumProduct) => {
                let i_v = a_v * u_next_v; // inner = a*u_next
                (1.0 - i_v, 1.0 - b_v, u_next_v, a_v)
            }
            // OUTER = prob-AND: f(b,i) = b*i;            df/db = i,    df/di = b
            // INNER = prob-OR:  g(a,u) = a+u-a*u;        dg/da = 1-u,  dg/du = 1-a
            (false, UntilSemantics::ProbSumProduct) => {
                let i_v = a_v + u_next_v - a_v * u_next_v; // inner = a OR u_next
                (i_v, b_v, 1.0 - u_next_v, 1.0 - a_v)
            }
            // OUTER = max: f(b,i) = max(b,i); indicator
            // INNER = min: g(a,u) = min(a,u); indicator
            (true, UntilSemantics::MaxMin) => {
                let i_v = a_v.min(u_next_v);
                let d_outer_d_b = if b_v >= i_v { 1.0 } else { 0.0 };
                let d_outer_d_i = if b_v < i_v { 1.0 } else { 0.0 };
                let d_inner_d_a = if a_v <= u_next_v { 1.0 } else { 0.0 };
                let d_inner_d_u = if u_next_v < a_v { 1.0 } else { 0.0 };
                (d_outer_d_b, d_outer_d_i, d_inner_d_a, d_inner_d_u)
            }
            // OUTER = min: f(b,i) = min(b,i); indicator
            // INNER = max: g(a,u) = max(a,u); indicator
            (false, UntilSemantics::MaxMin) => {
                let i_v = a_v.max(u_next_v);
                let d_outer_d_b = if b_v <= i_v { 1.0 } else { 0.0 };
                let d_outer_d_i = if b_v > i_v { 1.0 } else { 0.0 };
                let d_inner_d_a = if a_v >= u_next_v { 1.0 } else { 0.0 };
                let d_inner_d_u = if u_next_v > a_v { 1.0 } else { 0.0 };
                (d_outer_d_b, d_outer_d_i, d_inner_d_a, d_inner_d_u)
            }
        }
    };

    // Forward-in-time adjoint sweep: k = 0 .. T-2
    for k in 0..(t - 1) {
        let a_k = a.index_axis(Axis(axis), k);
        let b_k = b.index_axis(Axis(axis), k);
        let u_next = u.index_axis(Axis(axis), k + 1);
        let s_k = s.index_axis(Axis(axis), k).to_owned();

        // Compute element-wise: da contribution, db contribution, carry to s[k+1].
        let da_k = Zip::from(&b_k)
            .and(&a_k)
            .and(&u_next)
            .map_collect(|&b_v, &a_v, &u_v| {
                let (_, d_outer_d_i, d_inner_d_a, _) = local_partials(b_v, a_v, u_v);
                d_outer_d_i * d_inner_d_a
            });

        let db_k = Zip::from(&b_k)
            .and(&a_k)
            .and(&u_next)
            .map_collect(|&b_v, &a_v, &u_v| {
                let (d_outer_d_b, _, _, _) = local_partials(b_v, a_v, u_v);
                d_outer_d_b
            });

        let ds_next = Zip::from(&b_k)
            .and(&a_k)
            .and(&u_next)
            .map_collect(|&b_v, &a_v, &u_v| {
                let (_, d_outer_d_i, _, d_inner_d_u) = local_partials(b_v, a_v, u_v);
                d_outer_d_i * d_inner_d_u
            });

        // grad_a[k] += s[k] * (d_outer_d_i * d_inner_d_a)
        {
            let mut ga_k = grad_a.index_axis_mut(Axis(axis), k);
            let contribution = Zip::from(&s_k)
                .and(&da_k)
                .map_collect(|&s_v, &da_v| s_v * da_v);
            ga_k.zip_mut_with(&contribution, |acc, &v| *acc += v);
        }

        // grad_b[k] += s[k] * d_outer_d_b
        {
            let mut gb_k = grad_b.index_axis_mut(Axis(axis), k);
            let contribution = Zip::from(&s_k)
                .and(&db_k)
                .map_collect(|&s_v, &db_v| s_v * db_v);
            gb_k.zip_mut_with(&contribution, |acc, &v| *acc += v);
        }

        // Carry adjoint forward: s[k+1] += s[k] * (d_outer_d_i * d_inner_d_u)
        {
            let carry = Zip::from(&s_k)
                .and(&ds_next)
                .map_collect(|&s_v, &du_v| s_v * du_v);
            let mut s_next_slice = s.index_axis_mut(Axis(axis), k + 1);
            s_next_slice.zip_mut_with(&carry, |acc, &v| *acc += v);
        }
    }

    // Base-case step: k = T-1.
    // u[T-1] = OUTER(b[T-1], INNER(a[T-1], boundary_val))
    {
        let a_last = a.index_axis(Axis(axis), t - 1);
        let b_last = b.index_axis(Axis(axis), t - 1);
        let s_last = s.index_axis(Axis(axis), t - 1).to_owned();

        let da_last = Zip::from(&b_last).and(&a_last).map_collect(|&b_v, &a_v| {
            let (_, d_outer_d_i, d_inner_d_a, _) = local_partials(b_v, a_v, boundary_val);
            d_outer_d_i * d_inner_d_a
        });

        let db_last = Zip::from(&b_last).and(&a_last).map_collect(|&b_v, &a_v| {
            let (d_outer_d_b, _, _, _) = local_partials(b_v, a_v, boundary_val);
            d_outer_d_b
        });

        // grad_a[T-1] += s[T-1] * da_contribution
        {
            let mut ga_last = grad_a.index_axis_mut(Axis(axis), t - 1);
            let contribution = Zip::from(&s_last)
                .and(&da_last)
                .map_collect(|&s_v, &da_v| s_v * da_v);
            ga_last.zip_mut_with(&contribution, |acc, &v| *acc += v);
        }

        // grad_b[T-1] += s[T-1] * db_contribution
        {
            let mut gb_last = grad_b.index_axis_mut(Axis(axis), t - 1);
            let contribution = Zip::from(&s_last)
                .and(&db_last)
                .map_collect(|&s_v, &db_v| s_v * db_v);
            gb_last.zip_mut_with(&contribution, |acc, &v| *acc += v);
        }
    }

    (grad_a, grad_b)
}

/// Evaluate `a Until b` via backward scan along `axis`.
///
/// Thin wrapper around [`temporal_binary_scan`] with `form = TemporalBinaryForm::Until`.
pub fn until_scan(
    a: &ArrayViewD<f64>,
    b: &ArrayViewD<f64>,
    axis: usize,
    sem: UntilSemantics,
) -> ArrayD<f64> {
    temporal_binary_scan(a, b, axis, TemporalBinaryForm::Until, sem)
}

/// Vector-Jacobian product (backward pass) for [`until_scan`].
///
/// Thin wrapper around [`temporal_binary_scan_vjp`] with `form = TemporalBinaryForm::Until`.
pub fn until_scan_vjp(
    a: &ArrayViewD<f64>,
    b: &ArrayViewD<f64>,
    grad_out: &ArrayViewD<f64>,
    axis: usize,
    sem: UntilSemantics,
) -> (ArrayD<f64>, ArrayD<f64>) {
    temporal_binary_scan_vjp(a, b, grad_out, axis, TemporalBinaryForm::Until, sem)
}

// ---------------------------------------------------------------------------
// VJP helpers
// ---------------------------------------------------------------------------

/// Vector-Jacobian product (backward pass) for [`shift_next`].
///
/// `shift_next` moves `x[k+1]` → `out[k]`, so `d_loss/d_x[k] = d_loss/d_out[k-1]`
/// for `k ≥ 1`, and `d_loss/d_x[0] = 0`.
///
/// Equivalently this is the *reverse* shift (shift backward in time).
pub fn shift_prev(grad_out: &ArrayViewD<f64>, axis: usize) -> ArrayD<f64> {
    let t = grad_out.len_of(Axis(axis));
    let mut grad_x = ArrayD::zeros(grad_out.raw_dim());

    if t <= 1 {
        return grad_x;
    }

    // grad_x[k] = grad_out[k-1]  for k = 1..T-1
    for k in 1..t {
        let src = grad_out.index_axis(Axis(axis), k - 1);
        let mut dst = grad_x.index_axis_mut(Axis(axis), k);
        dst.assign(&src);
    }
    // grad_x[0] stays zero.
    grad_x
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2, Array};

    // Helper: 1-D array → ArrayD along axis 0.
    fn vec_to_arrayd(data: &[f64]) -> ArrayD<f64> {
        arr1(data).into_dyn()
    }

    // Helper: assert two arrays are element-wise within `tol`.
    fn assert_close(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64, msg: &str) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "{}: shape mismatch {:?} vs {:?}",
            msg,
            a.shape(),
            b.shape()
        );
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!(
                (av - bv).abs() < tol,
                "{}: element differs by {} (got {}, expected {})",
                msg,
                (av - bv).abs(),
                av,
                bv
            );
        }
    }

    // -----------------------------------------------------------------------
    // shift_next
    // -----------------------------------------------------------------------

    #[test]
    fn test_shift_next_basic() {
        let x = vec_to_arrayd(&[0.2, 0.8, 0.5]);
        let out = shift_next(&x.view(), 0);
        let expected = vec_to_arrayd(&[0.8, 0.5, 0.0]);
        assert_close(&out, &expected, 1e-12, "shift_next basic");
    }

    #[test]
    fn test_shift_next_t1_edge_case() {
        let x = vec_to_arrayd(&[0.7]);
        let out = shift_next(&x.view(), 0);
        let expected = vec_to_arrayd(&[0.0]);
        assert_close(&out, &expected, 1e-12, "shift_next T=1");
    }

    // -----------------------------------------------------------------------
    // until_scan - MaxMin (via wrapper)
    // -----------------------------------------------------------------------

    #[test]
    fn test_until_scan_max_min() {
        // a = [0.9, 0.9, 0.9], b = [0.0, 0.0, 0.4]
        // Backward scan:
        //   u[2] = b[2] = 0.4 (boundary=0 → INNER(0.9,0)=0, OUTER(0.4,0)=0.4)
        //   u[1] = max(b[1], min(a[1], u[2])) = max(0.0, min(0.9, 0.4)) = 0.4
        //   u[0] = max(b[0], min(a[0], u[1])) = max(0.0, min(0.9, 0.4)) = 0.4
        let a = vec_to_arrayd(&[0.9, 0.9, 0.9]);
        let b = vec_to_arrayd(&[0.0, 0.0, 0.4]);
        let u = until_scan(&a.view(), &b.view(), 0, UntilSemantics::MaxMin);
        let expected = vec_to_arrayd(&[0.4, 0.4, 0.4]);
        assert_close(&u, &expected, 1e-12, "until_scan MaxMin");
    }

    // -----------------------------------------------------------------------
    // until_scan - ProbSumProduct (via wrapper)
    // -----------------------------------------------------------------------

    #[test]
    fn test_until_scan_prob_sum_product() {
        // a = [0.5, 0.5], b = [0.3, 0.4]
        // u[1] = b[1] = 0.4  (boundary=0 → INNER(0.5,0)=0, OUTER(0.4,0)=0.4)
        // u[0] = b[0] + a[0]*u[1] - b[0]*a[0]*u[1]
        //       = 0.3 + 0.5*0.4 - 0.3*0.5*0.4
        //       = 0.3 + 0.2 - 0.06 = 0.44
        let a = vec_to_arrayd(&[0.5, 0.5]);
        let b = vec_to_arrayd(&[0.3, 0.4]);
        let u = until_scan(&a.view(), &b.view(), 0, UntilSemantics::ProbSumProduct);
        let expected = vec_to_arrayd(&[0.44, 0.4]);
        assert_close(&u, &expected, 1e-12, "until_scan ProbSumProduct");
    }

    // -----------------------------------------------------------------------
    // rank-2 test: axis=1
    // -----------------------------------------------------------------------

    #[test]
    fn test_shift_next_rank2_axis1() {
        // shape [2, 3]: time axis = 1
        let data = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let x = data.into_dyn();
        let out = shift_next(&x.view(), 1);

        assert_eq!(out.shape(), x.shape(), "rank-2 output shape preserved");

        // out[:, 0] = x[:, 1]  => [2.0, 5.0]
        // out[:, 1] = x[:, 2]  => [3.0, 6.0]
        // out[:, 2] = 0         => [0.0, 0.0]
        let expected = arr2(&[[2.0, 3.0, 0.0], [5.0, 6.0, 0.0]]).into_dyn();
        assert_close(&out, &expected, 1e-12, "shift_next rank-2 axis=1");
    }

    #[test]
    fn test_until_scan_rank2_axis1() {
        // shape [2, 3]: time axis = 1
        let a = Array::from_elem((2, 3), 0.9_f64).into_dyn();
        let b_data = arr2(&[[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]]);
        let b = b_data.into_dyn();
        let u = until_scan(&a.view(), &b.view(), 1, UntilSemantics::MaxMin);
        assert_eq!(u.shape(), a.shape(), "until_scan rank-2 shape preserved");
        // Expected: each row = [0.4, 0.4, 0.4] (same as 1-D case)
        let expected = Array::from_elem((2, 3), 0.4_f64).into_dyn();
        assert_close(&u, &expected, 1e-12, "until_scan rank-2");
    }

    // -----------------------------------------------------------------------
    // shift_prev is inverse of shift_next (for interior slices)
    // -----------------------------------------------------------------------

    #[test]
    fn test_shift_prev_inverse_of_shift_next() {
        let x = vec_to_arrayd(&[0.1, 0.2, 0.3, 0.4]);
        let shifted = shift_next(&x.view(), 0); // [0.2, 0.3, 0.4, 0.0]

        let grad_x = shift_prev(&shifted.view(), 0);

        // g = [0.2, 0.3, 0.4, 0.0], shift_prev(g) = [0.0, 0.2, 0.3, 0.4]
        let expected = vec_to_arrayd(&[0.0, 0.2, 0.3, 0.4]);
        assert_close(&grad_x, &expected, 1e-12, "shift_prev inverse");
    }

    // -----------------------------------------------------------------------
    // Finite-difference VJP check: shift_next (linear, so exact)
    // -----------------------------------------------------------------------

    #[test]
    fn test_shift_next_vjp_finite_difference() {
        let x0 = vec_to_arrayd(&[0.3, 0.7, 0.5, 0.2]);
        let g_out = vec_to_arrayd(&[1.0, 2.0, 3.0, 4.0]);

        // Analytic VJP.
        let grad_analytic = shift_prev(&g_out.view(), 0);

        let n = x0.len();
        let eps = 1e-7;
        let mut grad_fd = ArrayD::zeros(x0.raw_dim());
        for j in 0..n {
            let mut xp = x0.clone();
            *xp.iter_mut().nth(j).expect("index in bounds") += eps;
            let mut xm = x0.clone();
            *xm.iter_mut().nth(j).expect("index in bounds") -= eps;
            let out_p = shift_next(&xp.view(), 0);
            let out_m = shift_next(&xm.view(), 0);
            let diff = (&out_p - &out_m) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = grad_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        assert_close(
            &grad_analytic,
            &grad_fd,
            1e-8,
            "shift_next VJP finite-difference",
        );
    }

    // -----------------------------------------------------------------------
    // Finite-difference VJP check: until_scan ProbSumProduct (smooth)
    // -----------------------------------------------------------------------

    #[test]
    fn test_until_scan_vjp_prob_sum_product_finite_difference() {
        let a0 = vec_to_arrayd(&[0.4, 0.6, 0.8]);
        let b0 = vec_to_arrayd(&[0.2, 0.5, 0.3]);
        let g_out = vec_to_arrayd(&[1.0, 1.0, 1.0]);

        let (grad_a_analytic, grad_b_analytic) = until_scan_vjp(
            &a0.view(),
            &b0.view(),
            &g_out.view(),
            0,
            UntilSemantics::ProbSumProduct,
        );

        let n = a0.len();
        let eps = 1e-5;
        let tol = 1e-4;

        // Finite-difference for grad_a.
        let mut grad_a_fd = ArrayD::zeros(a0.raw_dim());
        for j in 0..n {
            let mut ap = a0.clone();
            *ap.iter_mut().nth(j).expect("index in bounds") += eps;
            let mut am = a0.clone();
            *am.iter_mut().nth(j).expect("index in bounds") -= eps;
            let u_p = until_scan(&ap.view(), &b0.view(), 0, UntilSemantics::ProbSumProduct);
            let u_m = until_scan(&am.view(), &b0.view(), 0, UntilSemantics::ProbSumProduct);
            let diff = (&u_p - &u_m) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = grad_a_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        // Finite-difference for grad_b.
        let mut grad_b_fd = ArrayD::zeros(b0.raw_dim());
        for j in 0..n {
            let mut bp = b0.clone();
            *bp.iter_mut().nth(j).expect("index in bounds") += eps;
            let mut bm = b0.clone();
            *bm.iter_mut().nth(j).expect("index in bounds") -= eps;
            let u_p = until_scan(&a0.view(), &bp.view(), 0, UntilSemantics::ProbSumProduct);
            let u_m = until_scan(&a0.view(), &bm.view(), 0, UntilSemantics::ProbSumProduct);
            let diff = (&u_p - &u_m) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = grad_b_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        assert_close(
            &grad_a_analytic,
            &grad_a_fd,
            tol,
            "until_scan VJP grad_a finite-difference",
        );
        assert_close(
            &grad_b_analytic,
            &grad_b_fd,
            tol,
            "until_scan VJP grad_b finite-difference",
        );
    }

    // -----------------------------------------------------------------------
    // WeakUntil tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_weak_until_scan_maxmin_boundary_1() {
        // a = [1,1,1], b = [0,0,0]
        // WeakUntil MaxMin: boundary = 1.0
        // u[2] = max(b[2], min(a[2], 1.0)) = max(0, min(1,1)) = max(0,1) = 1
        // u[1] = max(b[1], min(a[1], u[2])) = max(0, min(1,1)) = 1
        // u[0] = max(b[0], min(a[0], u[1])) = max(0, min(1,1)) = 1
        let a = vec_to_arrayd(&[1.0, 1.0, 1.0]);
        let b = vec_to_arrayd(&[0.0, 0.0, 0.0]);
        let u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            0,
            TemporalBinaryForm::WeakUntil,
            UntilSemantics::MaxMin,
        );
        let expected = vec_to_arrayd(&[1.0, 1.0, 1.0]);
        assert_close(&u, &expected, 1e-12, "WeakUntil MaxMin boundary=1");
    }

    #[test]
    fn test_release_scan_maxmin() {
        // a = [0,0,1], b = [1,1,1]
        // Release MaxMin: OUTER=min, INNER=max, boundary=1.0
        // u[2] = min(b[2], max(a[2], 1.0)) = min(1, max(1,1)) = min(1,1) = 1
        // u[1] = min(b[1], max(a[1], u[2])) = min(1, max(0,1)) = min(1,1) = 1
        // u[0] = min(b[0], max(a[0], u[1])) = min(1, max(0,1)) = min(1,1) = 1
        let a = vec_to_arrayd(&[0.0, 0.0, 1.0]);
        let b = vec_to_arrayd(&[1.0, 1.0, 1.0]);
        let u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            0,
            TemporalBinaryForm::Release,
            UntilSemantics::MaxMin,
        );
        let expected = vec_to_arrayd(&[1.0, 1.0, 1.0]);
        assert_close(&u, &expected, 1e-12, "Release MaxMin b holds always");
    }

    #[test]
    fn test_strong_release_scan_maxmin() {
        // a = [0,1,0], b = [1,0,1]
        // StrongRelease MaxMin: OUTER=min, INNER=max, boundary=0.0
        // u[2] = min(b[2], max(a[2], 0.0)) = min(1, max(0,0)) = min(1,0) = 0
        // u[1] = min(b[1], max(a[1], u[2])) = min(0, max(1,0)) = min(0,1) = 0
        // u[0] = min(b[0], max(a[0], u[1])) = min(1, max(0,0)) = min(1,0) = 0
        let a = vec_to_arrayd(&[0.0, 1.0, 0.0]);
        let b = vec_to_arrayd(&[1.0, 0.0, 1.0]);
        let u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            0,
            TemporalBinaryForm::StrongRelease,
            UntilSemantics::MaxMin,
        );
        let expected = vec_to_arrayd(&[0.0, 0.0, 0.0]);
        assert_close(&u, &expected, 1e-12, "StrongRelease MaxMin hand-computed");
    }

    #[test]
    fn test_weak_until_prob_sum_product() {
        // a = [0.5, 0.8], b = [0.2, 0.3]
        // WeakUntil ProbSumProduct: OUTER=OR, INNER=AND, boundary=1.0
        // u[1] = b[1] OR (a[1] AND 1.0)
        //      = OR(0.3, AND(0.8, 1.0)) = OR(0.3, 0.8)
        //      = 0.3 + 0.8 - 0.3*0.8 = 1.1 - 0.24 = 0.86
        // u[0] = OR(b[0], AND(a[0], u[1]))
        //      = OR(0.2, AND(0.5, 0.86))
        //      = OR(0.2, 0.43)
        //      = 0.2 + 0.43 - 0.2*0.43 = 0.63 - 0.086 = 0.544
        let a = vec_to_arrayd(&[0.5, 0.8]);
        let b = vec_to_arrayd(&[0.2, 0.3]);
        let u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            0,
            TemporalBinaryForm::WeakUntil,
            UntilSemantics::ProbSumProduct,
        );
        let u1 = 0.3_f64 + 0.8 - 0.3 * 0.8; // 0.86
        let inner0 = 0.5_f64 * u1; // 0.43
        let u0 = 0.2_f64 + inner0 - 0.2 * inner0; // 0.544
        let expected = vec_to_arrayd(&[u0, u1]);
        assert_close(
            &u,
            &expected,
            1e-12,
            "WeakUntil ProbSumProduct hand-computed",
        );
    }

    #[test]
    fn test_release_prob_sum_product() {
        // a = [0.4, 0.6], b = [0.7, 0.5]
        // Release ProbSumProduct: OUTER=AND(x,y)=x*y, INNER=OR(x,y)=x+y-xy, boundary=1.0
        // u[1] = AND(b[1], OR(a[1], 1.0))
        //      = b[1] * OR(0.6, 1.0)
        //      = 0.5 * (0.6 + 1.0 - 0.6) = 0.5 * 1.0 = 0.5
        // u[0] = AND(b[0], OR(a[0], u[1]))
        //      = b[0] * OR(0.4, 0.5)
        //      = 0.7 * (0.4 + 0.5 - 0.4*0.5)
        //      = 0.7 * (0.9 - 0.2) = 0.7 * 0.7 = 0.49
        let a = vec_to_arrayd(&[0.4, 0.6]);
        let b = vec_to_arrayd(&[0.7, 0.5]);
        let u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            0,
            TemporalBinaryForm::Release,
            UntilSemantics::ProbSumProduct,
        );
        let u1 = 0.5_f64 * (0.6_f64 + 1.0 - 0.6);
        let inner0 = 0.4_f64 + u1 - 0.4 * u1;
        let u0 = 0.7_f64 * inner0;
        let expected = vec_to_arrayd(&[u0, u1]);
        assert_close(&u, &expected, 1e-12, "Release ProbSumProduct hand-computed");
    }

    #[test]
    fn test_duality_release_vs_until() {
        // For Boolean-valued inputs: release(p,q,MaxMin) = 1 - until(1-p, 1-q, MaxMin)
        // Uses a = [0.0, 1.0, 0.0], b = [1.0, 1.0, 0.0] (Boolean values)
        let a = vec_to_arrayd(&[0.0, 1.0, 0.0]);
        let b = vec_to_arrayd(&[1.0, 1.0, 0.0]);

        let release_u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            0,
            TemporalBinaryForm::Release,
            UntilSemantics::MaxMin,
        );

        let one_minus_a = a.mapv(|v| 1.0 - v);
        let one_minus_b = b.mapv(|v| 1.0 - v);
        let until_u = until_scan(
            &one_minus_a.view(),
            &one_minus_b.view(),
            0,
            UntilSemantics::MaxMin,
        );
        let dual = until_u.mapv(|v| 1.0 - v);

        assert_close(
            &release_u,
            &dual,
            1e-12,
            "duality release vs until (Boolean MaxMin)",
        );
    }

    #[test]
    fn test_vjp_weak_until_finite_difference() {
        let a0 = vec_to_arrayd(&[0.4, 0.6, 0.7]);
        let b0 = vec_to_arrayd(&[0.2, 0.5, 0.3]);
        let g_out = vec_to_arrayd(&[1.0, 0.5, -0.3]);
        let sem = UntilSemantics::ProbSumProduct;
        let form = TemporalBinaryForm::WeakUntil;

        let (ga_analytic, gb_analytic) =
            temporal_binary_scan_vjp(&a0.view(), &b0.view(), &g_out.view(), 0, form, sem);

        let n = a0.len();
        let eps = 1e-5;
        let tol = 1e-4;

        let mut ga_fd = ArrayD::zeros(a0.raw_dim());
        for j in 0..n {
            let mut ap = a0.clone();
            *ap.iter_mut().nth(j).expect("j in bounds") += eps;
            let mut am = a0.clone();
            *am.iter_mut().nth(j).expect("j in bounds") -= eps;
            let up = temporal_binary_scan(&ap.view(), &b0.view(), 0, form, sem);
            let um = temporal_binary_scan(&am.view(), &b0.view(), 0, form, sem);
            let diff = (&up - &um) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = ga_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        let mut gb_fd = ArrayD::zeros(b0.raw_dim());
        for j in 0..n {
            let mut bp = b0.clone();
            *bp.iter_mut().nth(j).expect("j in bounds") += eps;
            let mut bm = b0.clone();
            *bm.iter_mut().nth(j).expect("j in bounds") -= eps;
            let up = temporal_binary_scan(&a0.view(), &bp.view(), 0, form, sem);
            let um = temporal_binary_scan(&a0.view(), &bm.view(), 0, form, sem);
            let diff = (&up - &um) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = gb_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        assert_close(
            &ga_analytic,
            &ga_fd,
            tol,
            "WeakUntil VJP grad_a finite-diff",
        );
        assert_close(
            &gb_analytic,
            &gb_fd,
            tol,
            "WeakUntil VJP grad_b finite-diff",
        );
    }

    #[test]
    fn test_vjp_release_finite_difference() {
        let a0 = vec_to_arrayd(&[0.3, 0.7, 0.5]);
        let b0 = vec_to_arrayd(&[0.6, 0.4, 0.8]);
        let g_out = vec_to_arrayd(&[0.5, -0.5, 1.0]);
        let sem = UntilSemantics::ProbSumProduct;
        let form = TemporalBinaryForm::Release;

        let (ga_analytic, gb_analytic) =
            temporal_binary_scan_vjp(&a0.view(), &b0.view(), &g_out.view(), 0, form, sem);

        let n = a0.len();
        let eps = 1e-5;
        let tol = 1e-4;

        let mut ga_fd = ArrayD::zeros(a0.raw_dim());
        for j in 0..n {
            let mut ap = a0.clone();
            *ap.iter_mut().nth(j).expect("j in bounds") += eps;
            let mut am = a0.clone();
            *am.iter_mut().nth(j).expect("j in bounds") -= eps;
            let up = temporal_binary_scan(&ap.view(), &b0.view(), 0, form, sem);
            let um = temporal_binary_scan(&am.view(), &b0.view(), 0, form, sem);
            let diff = (&up - &um) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = ga_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        let mut gb_fd = ArrayD::zeros(b0.raw_dim());
        for j in 0..n {
            let mut bp = b0.clone();
            *bp.iter_mut().nth(j).expect("j in bounds") += eps;
            let mut bm = b0.clone();
            *bm.iter_mut().nth(j).expect("j in bounds") -= eps;
            let up = temporal_binary_scan(&a0.view(), &bp.view(), 0, form, sem);
            let um = temporal_binary_scan(&a0.view(), &bm.view(), 0, form, sem);
            let diff = (&up - &um) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = gb_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        assert_close(&ga_analytic, &ga_fd, tol, "Release VJP grad_a finite-diff");
        assert_close(&gb_analytic, &gb_fd, tol, "Release VJP grad_b finite-diff");
    }

    #[test]
    fn test_vjp_strong_release_finite_difference() {
        let a0 = vec_to_arrayd(&[0.5, 0.4, 0.6]);
        let b0 = vec_to_arrayd(&[0.3, 0.7, 0.5]);
        let g_out = vec_to_arrayd(&[-0.5, 1.0, 0.5]);
        let sem = UntilSemantics::ProbSumProduct;
        let form = TemporalBinaryForm::StrongRelease;

        let (ga_analytic, gb_analytic) =
            temporal_binary_scan_vjp(&a0.view(), &b0.view(), &g_out.view(), 0, form, sem);

        let n = a0.len();
        let eps = 1e-5;
        let tol = 1e-4;

        let mut ga_fd = ArrayD::zeros(a0.raw_dim());
        for j in 0..n {
            let mut ap = a0.clone();
            *ap.iter_mut().nth(j).expect("j in bounds") += eps;
            let mut am = a0.clone();
            *am.iter_mut().nth(j).expect("j in bounds") -= eps;
            let up = temporal_binary_scan(&ap.view(), &b0.view(), 0, form, sem);
            let um = temporal_binary_scan(&am.view(), &b0.view(), 0, form, sem);
            let diff = (&up - &um) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = ga_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        let mut gb_fd = ArrayD::zeros(b0.raw_dim());
        for j in 0..n {
            let mut bp = b0.clone();
            *bp.iter_mut().nth(j).expect("j in bounds") += eps;
            let mut bm = b0.clone();
            *bm.iter_mut().nth(j).expect("j in bounds") -= eps;
            let up = temporal_binary_scan(&a0.view(), &bp.view(), 0, form, sem);
            let um = temporal_binary_scan(&a0.view(), &bm.view(), 0, form, sem);
            let diff = (&up - &um) / (2.0 * eps);
            let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
            if let Some(v) = gb_fd.iter_mut().nth(j) {
                *v = val;
            }
        }

        assert_close(
            &ga_analytic,
            &ga_fd,
            tol,
            "StrongRelease VJP grad_a finite-diff",
        );
        assert_close(
            &gb_analytic,
            &gb_fd,
            tol,
            "StrongRelease VJP grad_b finite-diff",
        );
    }

    #[test]
    fn test_binary_scan_rank2_axis1() {
        // WeakUntil on shape [2, 3], axis=1
        // Row 0: a=[0.9, 0.9, 0.9], b=[0.0, 0.0, 0.0]
        // Row 1: a=[0.5, 0.5, 0.5], b=[0.3, 0.3, 0.3]
        // WeakUntil MaxMin OUTER=max, INNER=min, boundary=1.0:
        // Row 0:
        //   u[2] = max(b[2], min(a[2], 1.0)) = max(0, min(0.9, 1)) = max(0, 0.9) = 0.9
        //   u[1] = max(b[1], min(a[1], u[2])) = max(0, min(0.9, 0.9)) = max(0, 0.9) = 0.9
        //   u[0] = max(b[0], min(a[0], u[1])) = max(0, min(0.9, 0.9)) = max(0, 0.9) = 0.9
        // Row 1:
        //   u[2] = max(0.3, min(0.5, 1.0)) = max(0.3, 0.5) = 0.5
        //   u[1] = max(0.3, min(0.5, 0.5)) = max(0.3, 0.5) = 0.5
        //   u[0] = max(0.3, min(0.5, 0.5)) = max(0.3, 0.5) = 0.5
        let a_data = arr2(&[[0.9, 0.9, 0.9], [0.5, 0.5, 0.5]]);
        let b_data = arr2(&[[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]]);
        let a = a_data.into_dyn();
        let b = b_data.into_dyn();

        let u = temporal_binary_scan(
            &a.view(),
            &b.view(),
            1,
            TemporalBinaryForm::WeakUntil,
            UntilSemantics::MaxMin,
        );

        assert_eq!(u.shape(), a.shape(), "rank-2 WeakUntil shape preserved");

        let expected = arr2(&[[0.9, 0.9, 0.9], [0.5, 0.5, 0.5]]).into_dyn();
        assert_close(&u, &expected, 1e-12, "WeakUntil rank-2 axis=1");
    }
}
