//! Kepler's third law: T^2 = a^3, i.e. T = a^(3/2).
//! Target budget (design §4.1): 15 s. Data: synthetic (semi-major axis a, period T).

use phop_core::Config;
use phop_examples::{kepler_dataset, report};

fn main() {
    let ds = kepler_dataset();
    let cfg = Config::default().max_depth(3).max_epochs(400).top_k(5);
    report("Kepler's third law", "T = a^(3/2)  (T^2 = a^3)", &ds, cfg);
}
