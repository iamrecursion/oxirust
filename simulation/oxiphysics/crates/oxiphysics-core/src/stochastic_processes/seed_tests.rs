//! Seeding of the internal linear congruential generator: the streams of
//! consecutive seeds must be uncorrelated.

use super::Lcg;

#[test]
fn consecutive_seeds_give_uncorrelated_streams() {
    // For a LCG used raw (or merely offset by one) the stream of seed `s + 1` is
    // the stream of seed `s` plus `a^k (mod 2^64)` at step `k`: the uniforms of
    // neighbouring seeds were correlated at -0.36, -0.45, 0.75 and 0.89 at steps
    // 1 to 4.  The standard error of an uncorrelated pair is 1 / sqrt(20000).
    let seeds = 20_000_u64;
    let first_seed = 12_345_u64;
    let table: Vec<[f64; 8]> = (0..=seeds)
        .map(|i| {
            let mut rng = Lcg::new(first_seed + i);
            std::array::from_fn(|_| rng.uniform())
        })
        .collect();
    let count = seeds as usize;
    for k in 0..8 {
        let current: Vec<f64> = table[..count].iter().map(|row| row[k]).collect();
        let next: Vec<f64> = table[1..].iter().map(|row| row[k]).collect();
        let r = crate::statistics::correlation(&current, &next);
        assert!(r.abs() < 0.05, "step {k}: correlation {r}");
    }
}
