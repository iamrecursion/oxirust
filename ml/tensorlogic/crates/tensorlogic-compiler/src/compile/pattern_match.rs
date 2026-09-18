//! Compilation of pattern-matching expressions.
//!
//! `TLExpr::Match { scrutinee, arms }` is lowered to a nest of `IfThenElse`
//! nodes at IR level (then compiled recursively), one for each non-wildcard
//! arm.  The last arm must be `MatchPattern::Wildcard` (enforced by validation).
//!
//! Lowering sketch:
//!
//! ```text
//! match s {
//!   :ok  => body0,
//!   1.0  => body1,
//!   _    => body2,
//! }
//! ```
//!
//! becomes:
//!
//! ```text
//! if (s == :ok)  then body0
//! else if (s == 1.0) then body1
//! else body2
//! ```

use anyhow::{anyhow, Result};
use tensorlogic_ir::{EinsumGraph, MatchPattern, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Compile a `TLExpr::Match` by lowering to nested `IfThenElse` and compiling
/// the resulting tree.
pub(crate) fn compile_match(
    scrutinee: &TLExpr,
    arms: &[(MatchPattern, Box<TLExpr>)],
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    if arms.is_empty() {
        return Err(anyhow!("Match expression has no arms"));
    }

    // Validation guarantees last arm is Wildcard; check it defensively.
    let last_pat = &arms[arms.len() - 1].0;
    if !matches!(last_pat, MatchPattern::Wildcard) {
        return Err(anyhow!(
            "Last arm of Match must be Wildcard — validate before compiling"
        ));
    }

    // Build a TLExpr tree of nested IfThenElse, then compile it.
    let lowered = lower_match_to_if_chain(scrutinee, arms)?;
    compile_expr(&lowered, ctx, graph)
}

/// Convert the match expression into a nested `IfThenElse` IR tree.
fn lower_match_to_if_chain(
    scrutinee: &TLExpr,
    arms: &[(MatchPattern, Box<TLExpr>)],
) -> Result<TLExpr> {
    // Work backwards: start with the wildcard body (last arm) and wrap each
    // preceding arm in an IfThenElse.
    let wildcard_body = arms
        .last()
        .ok_or_else(|| anyhow!("Empty arms in Match"))?
        .1
        .as_ref()
        .clone();

    // Build the chain from the penultimate arm back to the first.
    let non_wildcard = &arms[..arms.len() - 1];

    // Fold right: start with else-branch = wildcard_body.
    let mut chain = wildcard_body;
    for (pat, body) in non_wildcard.iter().rev() {
        let condition = pattern_condition(scrutinee, pat)?;
        chain = TLExpr::IfThenElse {
            condition: Box::new(condition),
            then_branch: Box::new(body.as_ref().clone()),
            else_branch: Box::new(chain),
        };
    }
    Ok(chain)
}

/// Build an equality condition: `scrutinee == <constant>`.
fn pattern_condition(scrutinee: &TLExpr, pat: &MatchPattern) -> Result<TLExpr> {
    let rhs = match pat {
        MatchPattern::ConstNumber(n) => TLExpr::Constant(*n),
        MatchPattern::ConstSymbol(s) => TLExpr::SymbolLiteral(s.clone()),
        MatchPattern::Wildcard => {
            return Err(anyhow!("Wildcard pattern in non-tail position is invalid"));
        }
    };
    Ok(TLExpr::Eq(Box::new(scrutinee.clone()), Box::new(rhs)))
}

/// Compile a `TLExpr::SymbolLiteral` as a named constant tensor.
pub(crate) fn compile_symbol_literal(
    symbol: &str,
    _ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Symbol literals are represented as named scalar tensors.
    // The backend is responsible for implementing symbol equality.
    let tensor_name = format!("sym_{symbol}");
    let tensor_idx = graph.add_tensor(&tensor_name);
    Ok(CompileState {
        tensor_idx,
        axes: String::new(),
    })
}
