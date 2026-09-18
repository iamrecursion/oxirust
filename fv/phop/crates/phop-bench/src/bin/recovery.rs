//! Symbolic-recovery benchmark runner.
//!
//! Runs the [`phop_bench::feynman_like_suite`] through the discoverer with a reasonable
//! shallow-depth config, then prints an aligned results table, a summary line, and a
//! LaTeX table (for `phop.tex` §Experiments).

use phop_bench::{results_to_latex_table, run_suite, BenchResult};
use phop_core::Config;

/// Recovery tolerance: a case counts as recovered when the best MSE drops below this.
const TOL: f64 = 1e-3;

fn main() {
    let cfg = Config::default()
        .max_depth(3)
        .max_epochs(400)
        .top_k(5)
        .seed(42);

    let results = run_suite(&cfg, TOL);
    print_table(&results);

    let recovered = results.iter().filter(|r| r.recovered).count();
    println!();
    println!("recovered {recovered} / {}", results.len());

    // Emit the LaTeX table for the paper's experiments section, clearly delimited so it can
    // be copy-pasted (or `sed`-extracted) without the human-readable noise above.
    println!();
    println!("===== BEGIN LATEX TABLE =====");
    print!("{}", results_to_latex_table(&results));
    println!("===== END LATEX TABLE =====");
}

/// Print the results as an aligned table.
fn print_table(results: &[BenchResult]) {
    let name_w = results
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(4)
        .max("name".len());
    let latex_w = results
        .iter()
        .map(|r| r.best_latex.len())
        .max()
        .unwrap_or(5)
        .max("latex".len());

    println!(
        "{:<name_w$}  {:<9}  {:>12}  {:>8}  {:>10}  {:>8}  {:<latex_w$}",
        "name", "recovered", "mse", "r2", "complexity", "ms", "latex",
    );
    println!(
        "{}  {}  {}  {}  {}  {}  {}",
        "-".repeat(name_w),
        "-".repeat(9),
        "-".repeat(12),
        "-".repeat(8),
        "-".repeat(10),
        "-".repeat(8),
        "-".repeat(latex_w),
    );
    for r in results {
        println!(
            "{:<name_w$}  {:<9}  {:>12.3e}  {:>8.4}  {:>10}  {:>8}  {:<latex_w$}",
            r.name,
            if r.recovered { "yes" } else { "no" },
            r.best_mse,
            r.r2,
            r.best_complexity,
            r.millis,
            r.best_latex,
        );
    }
}
