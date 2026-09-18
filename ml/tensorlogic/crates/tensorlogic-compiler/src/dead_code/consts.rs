//! Helper predicates recognising boolean-constant values encoded as `TLExpr::Constant`.

use tensorlogic_ir::TLExpr;

/// Returns `true` if `expr` is exactly `Constant(1.0)` (logical True).
#[inline]
pub(super) fn is_true_const(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Constant(v) if *v == 1.0)
}

/// Returns `true` if `expr` is exactly `Constant(0.0)` (logical False).
#[inline]
pub(super) fn is_false_const(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Constant(v) if *v == 0.0)
}
