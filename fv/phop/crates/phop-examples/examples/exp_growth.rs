//! Exponential growth: y = exp(x). A law that phop recovers *exactly* at depth 1
//! (`eml(x, 1) = exp(x) - ln(1) = exp(x)`) — the canonical "it works" demo for the README.
//! Data: synthetic (50 samples of `x` in `[0, 4)`).

use phop_core::Config;
use phop_examples::{exp_growth_dataset, report};

fn main() {
    let ds = exp_growth_dataset();
    let cfg = Config::default().max_depth(2).max_epochs(400).top_k(5);
    report("Exponential growth", "y = exp(x)", &ds, cfg);
}
