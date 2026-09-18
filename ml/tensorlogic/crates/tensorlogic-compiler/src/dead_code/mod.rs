//! Dead-code elimination pass.
//!
//! Submodules:
//! - `types` — public config, stats, and the main [`DeadCodeEliminator`] type.
//! - `consts` — helpers to detect constant truth/falsity.
//! - `fold` — constant folding for Boolean connectives and `if` branches.
//! - `free_vars` — free-variable analysis for `let`-binding elimination.
//! - `node_count` — AST node counting plus generic unary/binary recursion helpers.
//! - `eliminate_flow` — elimination arms for control-flow-shaped nodes.
//! - `eliminate_ops` — elimination arms for arithmetic / comparison / modal ops.
//! - `eliminate_ext` — elimination arms for fuzzy / set / aggregate / leaf nodes.
//! - `eliminate` — the core recursive dispatcher that delegates to the above.

mod consts;
mod eliminate;
mod eliminate_ext;
mod eliminate_flow;
mod eliminate_ops;
mod fold;
mod free_vars;
mod node_count;
mod types;

#[cfg(test)]
mod tests;

pub use types::{DceConfig, DceStats, DeadCodeEliminator};
