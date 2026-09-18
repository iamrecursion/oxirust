//! Michaelis-Menten kinetics: v = V_max * [S] / (K_m + [S]).
//! Target budget (design §4.1): 10 s. Data: synthetic (substrate [S], rate v; V_max=2, K_m=0.5).

use phop_core::Config;
use phop_examples::{michaelis_menten_dataset, report};

fn main() {
    let ds = michaelis_menten_dataset();
    let cfg = Config::default().max_depth(3).max_epochs(400).top_k(5);
    report(
        "Michaelis-Menten kinetics",
        "v = V_max * [S] / (K_m + [S])  (V_max=2, K_m=0.5)",
        &ds,
        cfg,
    );
}
