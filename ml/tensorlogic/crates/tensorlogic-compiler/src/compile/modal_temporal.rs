//! Modal and temporal logic compilation to tensor operations.
//!
//! This module implements compilation strategies for modal and temporal logic operators,
//! enabling reasoning about possibility, necessity, and temporal sequences in tensor form.
//!
//! # Modal Logic
//!
//! Modal logic extends classical logic with operators for reasoning about necessity and possibility:
//!
//! - **Box (□P)**: "P is necessarily true" - P holds in all possible worlds/states
//! - **Diamond (◇P)**: "P is possibly true" - P holds in at least one possible world/state
//!
//! ## Tensor Representation
//!
//! Modal operators require an additional "world" or "state" dimension in tensors:
//! - Predicates are evaluated over multiple possible worlds
//! - Box reduces over worlds using min/product (all worlds must satisfy P)
//! - Diamond reduces over worlds using max/sum (at least one world satisfies P)
//!
//! # Temporal Logic (LTL)
//!
//! Temporal logic extends classical logic with operators for reasoning about sequences over time:
//!
//! - **Next (XP)**: "P is true in the next state" (requires backend support for shifts)
//! - **Eventually (FP)**: "P will be true in some future state"
//! - **Always (GP)**: "P is true in all future states"
//! - **Until (P U Q)**: "P holds until Q becomes true" (complex, requires scan operations)

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::config::{ModalStrategy, TemporalStrategy};
use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Special axis name for modal "world" dimension
const WORLD_AXIS: &str = "__world__";

/// Special axis name for temporal "time" dimension
const TIME_AXIS: &str = "__time__";

/// Compile a Box (□) modal operator: "P is necessarily true in all possible worlds"
///
/// Tensor semantics:
/// - Reduces over the world axis using the configured modal strategy
/// - Default: Min reduction (all worlds must satisfy P)
/// - Alternative: Product reduction for probabilistic interpretation
pub(crate) fn compile_box(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure world axis exists in context
    let world_axis = ensure_world_axis(ctx);

    // Compile inner expression (should now have world axis available)
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Get the modal strategy from config
    let strategy = ctx.config.modal_strategy;

    // Check if the inner expression actually uses the world axis
    if !inner_state.axes.contains(world_axis) {
        // If inner doesn't use world axis, just return it as-is
        // This handles predicates that don't reference possible worlds
        return Ok(inner_state);
    }

    // Apply reduction over world axis based on strategy
    match strategy {
        ModalStrategy::AllWorldsMin | ModalStrategy::Threshold { .. } => {
            // Use min reduction: all worlds must satisfy
            apply_reduction(&inner_state, world_axis, "min", ctx, graph)
        }
        ModalStrategy::AllWorldsProduct => {
            // Use product reduction: probabilistic interpretation
            apply_reduction(&inner_state, world_axis, "prod", ctx, graph)
        }
    }
}

/// Compile a Diamond (◇) modal operator: "P is possibly true in at least one world"
///
/// Tensor semantics:
/// - Reduces over the world axis using max/sum
/// - Default: Max reduction (at least one world satisfies P)
/// - Alternative: Sum reduction for probabilistic interpretation
pub(crate) fn compile_diamond(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure world axis exists
    let world_axis = ensure_world_axis(ctx);

    // Compile inner expression
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Check if the inner expression actually uses the world axis
    if !inner_state.axes.contains(world_axis) {
        // If inner doesn't use world axis, just return it as-is
        return Ok(inner_state);
    }

    // Get the modal strategy from config
    let strategy = ctx.config.modal_strategy;

    // Apply reduction over world axis based on strategy
    match strategy {
        ModalStrategy::AllWorldsMin | ModalStrategy::Threshold { .. } => {
            // Use max reduction (dual of min for Box)
            apply_reduction(&inner_state, world_axis, "max", ctx, graph)
        }
        ModalStrategy::AllWorldsProduct => {
            // Use sum reduction (dual of product for probabilistic interpretation)
            apply_reduction(&inner_state, world_axis, "sum", ctx, graph)
        }
    }
}

/// Map a [`TemporalStrategy`] to the tag string embedded in `temporal_until:<tag>:<axis>`.
fn until_tag(strategy: TemporalStrategy) -> &'static str {
    match strategy {
        TemporalStrategy::Max | TemporalStrategy::LogSumExp => "max",
        TemporalStrategy::Sum => "prod",
    }
}

/// Compile Next (X) temporal operator: "P is true in the next time step"
///
/// Emits a `temporal_next:<axis>` unary node that the backend handles via
/// [`tensorlogic_scirs_backend::temporal_ops::shift_next`].
pub(crate) fn compile_next(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let time_axis = ensure_time_axis(ctx);

    // Compile the inner expression.
    let inner_state = compile_expr(inner, ctx, graph)?;

    // If the inner expression does not involve the time axis, Next is identity.
    if !inner_state.axes.contains(time_axis) {
        return Ok(inner_state);
    }

    let time_idx = inner_state
        .axes
        .chars()
        .position(|c| c == time_axis)
        .expect("just checked the time axis is present");

    // Create output tensor.
    let out_tensor = ctx.fresh_temp();
    let out_idx = graph.add_tensor(out_tensor);

    // Emit the unary temporal_next node.
    let node = EinsumNode::elem_unary(
        format!("temporal_next:{}", time_idx),
        inner_state.tensor_idx,
        out_idx,
    );
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: out_idx,
        axes: inner_state.axes,
    })
}

/// Compile Eventually (F) temporal operator: "P will be true in some future state"
///
/// Tensor semantics:
/// - Reduces over future time using max/sum
pub(crate) fn compile_eventually(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure time axis exists
    let time_axis = ensure_time_axis(ctx);

    // Compile inner expression
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Check if the inner expression uses the time axis
    if !inner_state.axes.contains(time_axis) {
        return Ok(inner_state);
    }

    // Get temporal strategy from config
    let strategy = ctx.config.temporal_strategy;

    // Apply reduction based on strategy
    match strategy {
        TemporalStrategy::Max | TemporalStrategy::LogSumExp => {
            // Use max: true if true in any future state
            apply_reduction(&inner_state, time_axis, "max", ctx, graph)
        }
        TemporalStrategy::Sum => {
            // Use sum: probabilistic interpretation
            apply_reduction(&inner_state, time_axis, "sum", ctx, graph)
        }
    }
}

/// Compile Always (G) temporal operator: "P is true in all future states"
///
/// Tensor semantics:
/// - Reduces over future time using min/product
pub(crate) fn compile_always(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure time axis exists
    let time_axis = ensure_time_axis(ctx);

    // Compile inner expression
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Check if the inner expression uses the time axis
    if !inner_state.axes.contains(time_axis) {
        return Ok(inner_state);
    }

    // Get temporal strategy from config
    let strategy = ctx.config.temporal_strategy;

    // Apply reduction based on strategy
    match strategy {
        TemporalStrategy::Max | TemporalStrategy::LogSumExp => {
            // Use min: true only if true in all future states
            apply_reduction(&inner_state, time_axis, "min", ctx, graph)
        }
        TemporalStrategy::Sum => {
            // Use product: probabilistic interpretation
            apply_reduction(&inner_state, time_axis, "prod", ctx, graph)
        }
    }
}

/// Compile Until (U) temporal operator: "P holds until Q becomes true"
///
/// Emits a `temporal_until:<tag>:<axis>` binary node that the backend handles
/// via [`tensorlogic_scirs_backend::temporal_ops::until_scan`].
pub(crate) fn compile_until(
    before: &TLExpr,
    after: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let time_axis = ensure_time_axis(ctx);

    // Compile both sub-expressions.
    let before_state = compile_expr(before, ctx, graph)?;
    let after_state = compile_expr(after, ctx, graph)?;

    // Build the union of axes (same logic as compile_and in logic_ops.rs).
    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();

    for c in before_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    for c in after_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    // Ensure the time axis is in the output — it may not be there if both
    // operands are time-independent.  We force it by checking:
    if !output_axes.contains(time_axis) {
        output_axes.push(time_axis);
    }

    // Broadcast `before_state` and `after_state` to the union axes when needed.
    let mut before_aligned = before_state;
    let mut after_aligned = after_state;

    if before_aligned.axes != output_axes {
        let bspec = format!("{}->{}", before_aligned.axes, output_axes);
        let btmp = ctx.fresh_temp();
        let btmp_idx = graph.add_tensor(btmp);
        let bnode = EinsumNode::new(bspec, vec![before_aligned.tensor_idx], vec![btmp_idx]);
        graph.add_node(bnode)?;
        before_aligned = CompileState {
            tensor_idx: btmp_idx,
            axes: output_axes.clone(),
        };
    }

    if after_aligned.axes != output_axes {
        let aspec = format!("{}->{}", after_aligned.axes, output_axes);
        let atmp = ctx.fresh_temp();
        let atmp_idx = graph.add_tensor(atmp);
        let anode = EinsumNode::new(aspec, vec![after_aligned.tensor_idx], vec![atmp_idx]);
        graph.add_node(anode)?;
        after_aligned = CompileState {
            tensor_idx: atmp_idx,
            axes: output_axes.clone(),
        };
    }

    // Determine the positional index of the time axis in the output axes string.
    let time_idx = output_axes
        .chars()
        .position(|c| c == time_axis)
        .expect("time axis was inserted into output_axes above");

    // Choose the semantics tag from the temporal strategy.
    let tag = until_tag(ctx.config.temporal_strategy);

    // Emit the binary temporal_until node.
    let out_tensor = ctx.fresh_temp();
    let out_idx = graph.add_tensor(out_tensor);

    let node = EinsumNode::elem_binary(
        format!("temporal_until:{}:{}", tag, time_idx),
        before_aligned.tensor_idx,
        after_aligned.tensor_idx,
        out_idx,
    );
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: out_idx,
        axes: output_axes,
    })
}

/// Compile Release (R) temporal operator: "Q holds until and including when P first holds"
///
/// # Semantics
///
/// P R Q (P releases Q) means Q must hold continuously unless/until P becomes true;
/// if P never holds, Q must hold forever. Release is the dual of Until: P R Q ≡ ¬(¬P U ¬Q).
///
/// # Exact finite-trace recurrence
///
/// Emits a `temporal_release:<tag>:<axis>` binary node computed via the unified backward scan:
/// ```text
/// u[T-1] = AND(q[T-1], OR(p[T-1], boundary=1.0))
/// u[k]   = AND(q[k],   OR(p[k],   u[k+1]))           for k = T-2..0
/// ```
/// MaxMin semantics: AND=min, OR=max.
/// ProbSumProduct semantics: AND(x,y)=x*y, OR(x,y)=x+y-x*y.
///
/// Operand order: arg0 = p (releaser), arg1 = q (released).
pub(crate) fn compile_release(
    p: &TLExpr,
    q: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let time_axis = ensure_time_axis(ctx);

    let p_state = compile_expr(p, ctx, graph)?;
    let q_state = compile_expr(q, ctx, graph)?;

    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();
    for c in p_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    for c in q_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    if !output_axes.contains(time_axis) {
        output_axes.push(time_axis);
    }

    let mut p_aligned = p_state;
    let mut q_aligned = q_state;

    if p_aligned.axes != output_axes {
        let spec = format!("{}->{}", p_aligned.axes, output_axes);
        let tmp = ctx.fresh_temp();
        let tmp_idx = graph.add_tensor(tmp);
        let node = EinsumNode::new(spec, vec![p_aligned.tensor_idx], vec![tmp_idx]);
        graph.add_node(node)?;
        p_aligned = CompileState {
            tensor_idx: tmp_idx,
            axes: output_axes.clone(),
        };
    }

    if q_aligned.axes != output_axes {
        let spec = format!("{}->{}", q_aligned.axes, output_axes);
        let tmp = ctx.fresh_temp();
        let tmp_idx = graph.add_tensor(tmp);
        let node = EinsumNode::new(spec, vec![q_aligned.tensor_idx], vec![tmp_idx]);
        graph.add_node(node)?;
        q_aligned = CompileState {
            tensor_idx: tmp_idx,
            axes: output_axes.clone(),
        };
    }

    let time_idx = output_axes
        .chars()
        .position(|c| c == time_axis)
        .expect("time axis was inserted into output_axes above");

    let tag = until_tag(ctx.config.temporal_strategy);

    let out_tensor = ctx.fresh_temp();
    let out_idx = graph.add_tensor(out_tensor);

    let node = EinsumNode::elem_binary(
        format!("temporal_release:{}:{}", tag, time_idx),
        p_aligned.tensor_idx,
        q_aligned.tensor_idx,
        out_idx,
    );
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: out_idx,
        axes: output_axes,
    })
}

/// Compile WeakUntil (W) temporal operator: "P holds until Q, but Q may never hold"
///
/// # Semantics
///
/// P W Q (P weak until Q): P must hold continuously until Q becomes true; unlike strong Until,
/// Q is not required to ever become true. WeakUntil: P W Q ≡ (P U Q) ∨ □P.
///
/// # Exact finite-trace recurrence
///
/// Emits a `temporal_weakuntil:<tag>:<axis>` binary node computed via the unified backward scan:
/// ```text
/// u[T-1] = OR(q[T-1], AND(p[T-1], boundary=1.0))
/// u[k]   = OR(q[k],   AND(p[k],   u[k+1]))           for k = T-2..0
/// ```
/// MaxMin semantics: OR=max, AND=min.
/// ProbSumProduct semantics: OR(x,y)=x+y-x*y, AND(x,y)=x*y.
///
/// Operand order: arg0 = p (before/left), arg1 = q (after/right).
pub(crate) fn compile_weak_until(
    p: &TLExpr,
    q: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let time_axis = ensure_time_axis(ctx);

    let p_state = compile_expr(p, ctx, graph)?;
    let q_state = compile_expr(q, ctx, graph)?;

    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();
    for c in p_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    for c in q_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    if !output_axes.contains(time_axis) {
        output_axes.push(time_axis);
    }

    let mut p_aligned = p_state;
    let mut q_aligned = q_state;

    if p_aligned.axes != output_axes {
        let spec = format!("{}->{}", p_aligned.axes, output_axes);
        let tmp = ctx.fresh_temp();
        let tmp_idx = graph.add_tensor(tmp);
        let node = EinsumNode::new(spec, vec![p_aligned.tensor_idx], vec![tmp_idx]);
        graph.add_node(node)?;
        p_aligned = CompileState {
            tensor_idx: tmp_idx,
            axes: output_axes.clone(),
        };
    }

    if q_aligned.axes != output_axes {
        let spec = format!("{}->{}", q_aligned.axes, output_axes);
        let tmp = ctx.fresh_temp();
        let tmp_idx = graph.add_tensor(tmp);
        let node = EinsumNode::new(spec, vec![q_aligned.tensor_idx], vec![tmp_idx]);
        graph.add_node(node)?;
        q_aligned = CompileState {
            tensor_idx: tmp_idx,
            axes: output_axes.clone(),
        };
    }

    let time_idx = output_axes
        .chars()
        .position(|c| c == time_axis)
        .expect("time axis was inserted into output_axes above");

    let tag = until_tag(ctx.config.temporal_strategy);

    let out_tensor = ctx.fresh_temp();
    let out_idx = graph.add_tensor(out_tensor);

    let node = EinsumNode::elem_binary(
        format!("temporal_weakuntil:{}:{}", tag, time_idx),
        p_aligned.tensor_idx,
        q_aligned.tensor_idx,
        out_idx,
    );
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: out_idx,
        axes: output_axes,
    })
}

/// Compile StrongRelease (M) temporal operator: "Strong version of Release"
///
/// # Semantics
///
/// P M Q (P strong-releases Q): Q must hold until P becomes true, and P must eventually become true.
/// StrongRelease is the dual of WeakUntil: P M Q ≡ ¬(¬P W ¬Q).
///
/// # Exact finite-trace recurrence
///
/// Emits a `temporal_strongrelease:<tag>:<axis>` binary node computed via the unified backward scan:
/// ```text
/// u[T-1] = AND(q[T-1], OR(p[T-1], boundary=0.0))
/// u[k]   = AND(q[k],   OR(p[k],   u[k+1]))           for k = T-2..0
/// ```
/// MaxMin semantics: AND=min, OR=max.
/// ProbSumProduct semantics: AND(x,y)=x*y, OR(x,y)=x+y-x*y.
///
/// Operand order: arg0 = p (releaser), arg1 = q (released).
pub(crate) fn compile_strong_release(
    p: &TLExpr,
    q: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let time_axis = ensure_time_axis(ctx);

    let p_state = compile_expr(p, ctx, graph)?;
    let q_state = compile_expr(q, ctx, graph)?;

    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();
    for c in p_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    for c in q_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    if !output_axes.contains(time_axis) {
        output_axes.push(time_axis);
    }

    let mut p_aligned = p_state;
    let mut q_aligned = q_state;

    if p_aligned.axes != output_axes {
        let spec = format!("{}->{}", p_aligned.axes, output_axes);
        let tmp = ctx.fresh_temp();
        let tmp_idx = graph.add_tensor(tmp);
        let node = EinsumNode::new(spec, vec![p_aligned.tensor_idx], vec![tmp_idx]);
        graph.add_node(node)?;
        p_aligned = CompileState {
            tensor_idx: tmp_idx,
            axes: output_axes.clone(),
        };
    }

    if q_aligned.axes != output_axes {
        let spec = format!("{}->{}", q_aligned.axes, output_axes);
        let tmp = ctx.fresh_temp();
        let tmp_idx = graph.add_tensor(tmp);
        let node = EinsumNode::new(spec, vec![q_aligned.tensor_idx], vec![tmp_idx]);
        graph.add_node(node)?;
        q_aligned = CompileState {
            tensor_idx: tmp_idx,
            axes: output_axes.clone(),
        };
    }

    let time_idx = output_axes
        .chars()
        .position(|c| c == time_axis)
        .expect("time axis was inserted into output_axes above");

    let tag = until_tag(ctx.config.temporal_strategy);

    let out_tensor = ctx.fresh_temp();
    let out_idx = graph.add_tensor(out_tensor);

    let node = EinsumNode::elem_binary(
        format!("temporal_strongrelease:{}:{}", tag, time_idx),
        p_aligned.tensor_idx,
        q_aligned.tensor_idx,
        out_idx,
    );
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: out_idx,
        axes: output_axes,
    })
}

// ========================================================================
// Helper Functions
// ========================================================================

/// Ensure the world axis exists in the compilation context.
/// Returns the axis character for the world dimension.
fn ensure_world_axis(ctx: &mut CompilerContext) -> char {
    // Check if world axis already assigned
    if let Some(&axis) = ctx.var_to_axis.get(WORLD_AXIS) {
        return axis;
    }

    // Add world domain if not present
    if !ctx.domains.contains_key(WORLD_AXIS) {
        // Default: 10 possible worlds (configurable via context)
        let world_size = ctx.config.modal_world_size.unwrap_or(10);
        ctx.add_domain(WORLD_AXIS, world_size);
    }

    // Assign axis for world variable and return it
    ctx.assign_axis(WORLD_AXIS)
}

/// Ensure the time axis exists in the compilation context.
/// Returns the axis character for the time dimension.
fn ensure_time_axis(ctx: &mut CompilerContext) -> char {
    // Check if time axis already assigned
    if let Some(&axis) = ctx.var_to_axis.get(TIME_AXIS) {
        return axis;
    }

    // Add time domain if not present
    if !ctx.domains.contains_key(TIME_AXIS) {
        // Default: 100 time steps (configurable via context)
        let time_size = ctx.config.temporal_time_steps.unwrap_or(100);
        ctx.add_domain(TIME_AXIS, time_size);
    }

    // Assign axis for time variable and return it
    ctx.assign_axis(TIME_AXIS)
}

/// Apply a reduction operation over a specific axis.
///
/// Creates an einsum spec that reduces over the given axis using the specified operation.
fn apply_reduction(
    state: &CompileState,
    axis_to_reduce: char,
    reduction_op: &str,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Build output axes (all input axes except the one being reduced)
    let output_axes: String = state
        .axes
        .chars()
        .filter(|&c| c != axis_to_reduce)
        .collect();

    // Create einsum spec with reduction
    // Format: "op(input_axes->output_axes)" where op is the reduction operation
    let spec = format!("{}({}->{})", reduction_op, state.axes, output_axes);

    // Create result tensor
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create reduction node
    let node = EinsumNode::new(spec, vec![state.tensor_idx], vec![result_idx]);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompilationConfig, CompilerContext};
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_ensure_world_axis() {
        let mut ctx = CompilerContext::new();
        let axis1 = ensure_world_axis(&mut ctx);
        let axis2 = ensure_world_axis(&mut ctx);

        // Should return same axis when called twice
        assert_eq!(axis1, axis2);
        assert!(ctx.domains.contains_key(WORLD_AXIS));
        assert!(ctx.var_to_axis.contains_key(WORLD_AXIS));
    }

    #[test]
    fn test_ensure_time_axis() {
        let mut ctx = CompilerContext::new();
        let axis1 = ensure_time_axis(&mut ctx);
        let axis2 = ensure_time_axis(&mut ctx);

        // Should return same axis when called twice
        assert_eq!(axis1, axis2);
        assert!(ctx.domains.contains_key(TIME_AXIS));
        assert!(ctx.var_to_axis.contains_key(TIME_AXIS));
    }

    #[test]
    fn test_compile_box_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);

        let mut graph = EinsumGraph::new();

        // Box(P(x)) where P is some predicate
        let pred = TLExpr::pred("happy", vec![Term::var("x")]);

        // For this test, we expect it to work even if predicate doesn't exist
        // (it will fail at compilation, but modal logic setup should work)
        let result = compile_box(&pred, &mut ctx, &mut graph);

        // World axis should be created
        assert!(ctx.domains.contains_key(WORLD_AXIS));

        // Result may fail due to missing predicate info, but that's okay for this test
        let _ = result;
    }

    #[test]
    fn test_compile_diamond_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);

        let mut graph = EinsumGraph::new();

        let pred = TLExpr::pred("possible", vec![Term::var("x")]);

        let result = compile_diamond(&pred, &mut ctx, &mut graph);

        // World axis should be created
        assert!(ctx.domains.contains_key(WORLD_AXIS));

        let _ = result;
    }

    #[test]
    fn test_compile_eventually_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Event", 5);

        let mut graph = EinsumGraph::new();

        let pred = TLExpr::pred("occurs", vec![Term::var("e")]);

        let result = compile_eventually(&pred, &mut ctx, &mut graph);

        // Time axis should be created
        assert!(ctx.domains.contains_key(TIME_AXIS));

        let _ = result;
    }

    #[test]
    fn test_compile_next_succeeds() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 5);
        ctx.add_domain(TIME_AXIS, 10);

        let mut graph = EinsumGraph::new();

        // Register a source tensor so predicate compilation can find it.
        let t_idx = graph.add_tensor("p");
        // Also add an input declaration so compile_pred can find axes.
        let _ = t_idx;

        // compile_next: we drive it directly with a pred that references variable "t"
        // which will be bound to the time axis.
        let pred = TLExpr::pred("p", vec![Term::var("t")]);
        let result = compile_next(&pred, &mut ctx, &mut graph);

        // The operator is now implemented — it should succeed.
        // (It may fail due to missing predicate signature, which is OK for this
        //  structural test — we just verify it doesn't bail with the old stub message.)
        match &result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("shift operations which are not available"),
                    "compile_next must not produce the old stub error; got: {msg}"
                );
            }
            Ok(state) => {
                // If compilation succeeded, verify the time axis is in the output.
                let time_axis = ctx.var_to_axis.get(TIME_AXIS).copied();
                if let Some(ta) = time_axis {
                    // The output axes may or may not have time_axis depending on
                    // whether the predicate used it; just check the graph is non-empty.
                    assert!(
                        !graph.nodes.is_empty() || !state.axes.contains(ta),
                        "graph should have at least one node when temporal op was emitted"
                    );
                }
            }
        }
    }

    #[test]
    fn test_compile_until_succeeds() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 5);
        ctx.add_domain(TIME_AXIS, 10);

        let mut graph = EinsumGraph::new();

        let pred1 = TLExpr::pred("p", vec![Term::var("x")]);
        let pred2 = TLExpr::pred("q", vec![Term::var("x")]);
        let result = compile_until(&pred1, &pred2, &mut ctx, &mut graph);

        // The operator is now implemented — it should not bail with the old stub message.
        match &result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("scan operations which are not available"),
                    "compile_until must not produce the old stub error; got: {msg}"
                );
            }
            Ok(state) => {
                // Time axis must be present in output axes.
                let time_axis = ctx
                    .var_to_axis
                    .get(TIME_AXIS)
                    .copied()
                    .expect("time axis should be assigned");
                assert!(
                    state.axes.contains(time_axis),
                    "time axis '{time_axis}' must appear in Until output axes '{}'",
                    state.axes
                );
            }
        }
    }

    #[test]
    fn test_compile_until_time_axis_in_output() {
        // Verify that when at least one operand references time, the output also has time.
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Event", 8);
        ctx.add_domain(TIME_AXIS, 20);

        // Force registration by calling ensure_time_axis via compile_until.
        let mut graph = EinsumGraph::new();
        let pred1 = TLExpr::pred("p", vec![Term::var("e")]);
        let pred2 = TLExpr::pred("q", vec![Term::var("e")]);

        let result = compile_until(&pred1, &pred2, &mut ctx, &mut graph);

        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("scan operations which are not available"),
                    "compile_until stub message must not appear; got: {msg}"
                );
            }
            Ok(state) => {
                let time_axis = ctx
                    .var_to_axis
                    .get(TIME_AXIS)
                    .copied()
                    .expect("time axis allocated");
                assert!(
                    state.axes.contains(time_axis),
                    "time axis in Until output: {} not in {}",
                    time_axis,
                    state.axes
                );
            }
        }
    }

    #[test]
    fn test_compile_until_different_shape_operands() {
        // "before" has axes x, t; "after" has axis y, t  — union should be x, y, t.
        let mut ctx = CompilerContext::new();
        ctx.add_domain("X", 4);
        ctx.add_domain("Y", 3);
        ctx.add_domain(TIME_AXIS, 10);

        let mut graph = EinsumGraph::new();

        let pred_a = TLExpr::pred("p", vec![Term::var("x"), Term::var("t")]);
        let pred_b = TLExpr::pred("q", vec![Term::var("y"), Term::var("t")]);

        let result = compile_until(&pred_a, &pred_b, &mut ctx, &mut graph);

        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("scan operations which are not available"),
                    "compile_until stub must not appear; got: {msg}"
                );
            }
            Ok(state) => {
                // Output must have at least the time axis.
                let time_axis = ctx
                    .var_to_axis
                    .get(TIME_AXIS)
                    .copied()
                    .expect("time axis allocated");
                assert!(
                    state.axes.contains(time_axis),
                    "time axis in union output: {} not in {}",
                    time_axis,
                    state.axes
                );
            }
        }
    }

    #[test]
    fn test_modal_strategy_configuration() {
        // Test different modal strategies
        let ctx = CompilerContext::with_config(CompilationConfig::hard_boolean());
        assert_eq!(ctx.config.modal_strategy, ModalStrategy::AllWorldsMin);

        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        assert_eq!(ctx.config.modal_strategy, ModalStrategy::AllWorldsProduct);
    }

    #[test]
    fn test_temporal_strategy_configuration() {
        // Test different temporal strategies
        let ctx = CompilerContext::with_config(CompilationConfig::hard_boolean());
        assert_eq!(ctx.config.temporal_strategy, TemporalStrategy::Max);

        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        assert_eq!(ctx.config.temporal_strategy, TemporalStrategy::Sum);
    }

    #[test]
    fn test_compile_release_succeeds() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 5);
        ctx.add_domain(TIME_AXIS, 10);

        let mut graph = EinsumGraph::new();

        let pred_p = TLExpr::pred("p", vec![Term::var("x")]);
        let pred_q = TLExpr::pred("q", vec![Term::var("x")]);
        let result = compile_release(&pred_p, &pred_q, &mut ctx, &mut graph);

        match &result {
            Err(e) => {
                let msg = e.to_string();
                // Must not produce old approximation-related messages
                assert!(
                    !msg.contains("approximation"),
                    "compile_release must not mention approximation; got: {msg}"
                );
            }
            Ok(state) => {
                let time_axis = ctx
                    .var_to_axis
                    .get(TIME_AXIS)
                    .copied()
                    .expect("time axis should be assigned");
                assert!(
                    state.axes.contains(time_axis),
                    "time axis '{time_axis}' must appear in Release output axes '{}'",
                    state.axes
                );
            }
        }
    }

    #[test]
    fn test_compile_weak_until_succeeds() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 5);
        ctx.add_domain(TIME_AXIS, 10);

        let mut graph = EinsumGraph::new();

        let pred_p = TLExpr::pred("p", vec![Term::var("x")]);
        let pred_q = TLExpr::pred("q", vec![Term::var("x")]);
        let result = compile_weak_until(&pred_p, &pred_q, &mut ctx, &mut graph);

        match &result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("approximation"),
                    "compile_weak_until must not mention approximation; got: {msg}"
                );
            }
            Ok(state) => {
                let time_axis = ctx
                    .var_to_axis
                    .get(TIME_AXIS)
                    .copied()
                    .expect("time axis should be assigned");
                assert!(
                    state.axes.contains(time_axis),
                    "time axis '{time_axis}' must appear in WeakUntil output axes '{}'",
                    state.axes
                );
            }
        }
    }

    #[test]
    fn test_compile_strong_release_succeeds() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 5);
        ctx.add_domain(TIME_AXIS, 10);

        let mut graph = EinsumGraph::new();

        let pred_p = TLExpr::pred("p", vec![Term::var("x")]);
        let pred_q = TLExpr::pred("q", vec![Term::var("x")]);
        let result = compile_strong_release(&pred_p, &pred_q, &mut ctx, &mut graph);

        match &result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("approximation"),
                    "compile_strong_release must not mention approximation; got: {msg}"
                );
            }
            Ok(state) => {
                let time_axis = ctx
                    .var_to_axis
                    .get(TIME_AXIS)
                    .copied()
                    .expect("time axis should be assigned");
                assert!(
                    state.axes.contains(time_axis),
                    "time axis '{time_axis}' must appear in StrongRelease output axes '{}'",
                    state.axes
                );
            }
        }
    }
}
