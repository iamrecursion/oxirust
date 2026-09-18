//! Affine-leaf prototype vs baseline phop on the low-arity Feynman equations (M6 root-cause spike).
//!
//! For every `*.csv` in a directory with at most `max_vars` inputs, runs BOTH the baseline depth-3
//! discoverer and the affine-leaf prototype ([`phop_core::discover_affine`]) on identical data, and
//! prints their R² side by side. Quantifies how much "affine leaves + deeper search" actually buys.
//!
//! Usage: `cargo run -p phop-bench --release --bin affine_feynman -- <csv_dir> [max_vars]`

use phop_bench::r2_score;
use phop_core::{discover_affine_pareto, eval_tree, Config, DataSet, Discoverer};
use std::path::PathBuf;
use std::time::Instant;

const R2_TOL: f64 = 0.999;
/// Search bounds: up to 3 `eml` nodes (monomials/ratios are depth-1 with log-linear leaves, so this
/// is ample) over a bounded candidate pool of leaf-type assignments.
const MAX_INTERNAL: usize = 3;
const CAND_CAP: usize = 2000;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let max_vars = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2);
    // `affine_only` skips the (slow, OOM-at-9-vars, already-measured) baseline discoverer and runs
    // only the affine engine — used for the full-set rerun where baseline numbers are merged from
    // the `feynman` bin's output instead.
    let affine_only = std::env::args().nth(3).as_deref() == Some("affine_only");

    let mut paths: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "csv"))
            .collect(),
        Err(e) => {
            eprintln!("affine_feynman: cannot read dir {dir}: {e}");
            std::process::exit(1);
        }
    };
    paths.sort();

    let cfg = Config::default()
        .max_depth(3)
        .max_epochs(300)
        .top_k(5)
        .seed(42);

    println!("name,nvars,base_r2,affine_r2,base_rec,affine_rec,affine_symbolic,affine_nodes,affine_ms,affine_expr");
    let (mut base_rec, mut aff_rec, mut sym_rec, mut considered) = (0usize, 0usize, 0usize, 0usize);

    for p in &paths {
        let name = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        let Ok(ds) = DataSet::from_csv(p) else {
            continue;
        };
        if ds.n_vars() > max_vars {
            continue;
        }
        considered += 1;

        // Baseline depth-3 discoverer (skipped in affine_only mode).
        let base_r2 = if affine_only {
            f64::NAN
        } else {
            Discoverer::new(cfg.clone())
                .fit(&ds)
                .ok()
                .and_then(|f| f.best().cloned())
                .and_then(|b| {
                    eval_tree(&b.tree, &ds.x)
                        .ok()
                        .map(|pr| r2_score(&ds.y, &pr))
                })
                .unwrap_or(f64::NAN)
        };

        // Rich-leaf engine: take the Pareto front so a *simpler* member can supply the symbolic
        // (cleanly-snapped) recovery even when the min-MSE member is a flexible numeric fit.
        let start = Instant::now();
        let front = discover_affine_pareto(&ds.x, &ds.y, MAX_INTERNAL, CAND_CAP);
        let aff_ms = start.elapsed().as_millis();
        let best = front
            .iter()
            .max_by(|a, b| a.r2.partial_cmp(&b.r2).unwrap_or(std::cmp::Ordering::Equal));
        let (aff_r2, nodes, expr) = match best {
            Some(s) => (s.r2, s.nodes, s.expr.clone()),
            None => (f64::NAN, 0, String::new()),
        };
        // Symbolic recovery: some front member snaps to clean rational exponents AND keeps R² ≥ tol.
        let symbolic = front.iter().any(|s| s.symbolic && s.r2 >= R2_TOL);

        let base_ok = base_r2.is_finite() && base_r2 >= R2_TOL;
        let aff_ok = aff_r2.is_finite() && aff_r2 >= R2_TOL;
        base_rec += usize::from(base_ok);
        aff_rec += usize::from(aff_ok);
        sym_rec += usize::from(symbolic);

        println!(
            "{name},{},{base_r2:.4},{aff_r2:.4},{base_ok},{aff_ok},{symbolic},{nodes},{aff_ms},{}",
            ds.n_vars(),
            expr.replace(',', ";")
        );
    }

    eprintln!(
        "affine_feynman: over {considered} eqs (≤{max_vars} vars) — baseline {base_rec}, affine numeric {aff_rec}, affine SYMBOLIC {sym_rec} (R² ≥ {R2_TOL})"
    );
}
