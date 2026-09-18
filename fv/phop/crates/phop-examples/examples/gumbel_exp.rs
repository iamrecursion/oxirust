//! Differentiable topology demo (Layer B / M2): recover y = exp(x) by Gumbel-Softmax
//! leaf selection rather than structural enumeration. Each leaf learns, by gradient descent,
//! whether it should be the input variable or a constant.

use phop_core::{discover_gumbel, Config};
use phop_examples::dataset_1d;

fn main() {
    let xs: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.08).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| x.exp()).collect();
    let ds = dataset_1d(&xs, &ys);

    let cfg = Config::default()
        .max_depth(1)
        .population(6)
        .max_epochs(1200)
        .learning_rate(0.1)
        .top_k(5);

    println!("=== Gumbel-Softmax topology search ===");
    println!("target law: y = exp(x)\n");
    match discover_gumbel(&ds, &cfg) {
        Ok(front) => {
            for (i, s) in front.pareto_top(5).iter().enumerate() {
                println!(
                    "  #{}  complexity={:<3}  mse={:.4e}  {}",
                    i + 1,
                    s.complexity,
                    s.mse,
                    s.pretty()
                );
                println!("       latex: {}", s.latex());
            }
        }
        Err(e) => println!("  discovery failed: {e}"),
    }
}
