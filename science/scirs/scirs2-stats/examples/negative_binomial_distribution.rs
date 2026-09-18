use scirs2_stats::distributions::{nbinom, NegativeBinomial};
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Negative Binomial Distribution Example");
    println!("======================================\n");

    // Create a Negative Binomial distribution with r=5 and p=0.3
    let nb = nbinom(5.0f64, 0.3)?;

    // Alternatively, we can use the constructor directly
    let _nb_alt = NegativeBinomial::new(5.0f64, 0.3)?;

    // Display distribution information
    println!("Distribution Parameters:");
    println!("- r = 5 (number of successes to achieve)");
    println!("- p = 0.3 (success probability)\n");

    println!("Description: The Negative Binomial distribution models the number of failures");
    println!("before achieving r successes in independent Bernoulli trials with probability p.");
    println!("It's a generalization of the Geometric distribution (which is NB with r=1).\n");

    // Calculate and display PMF values
    println!("Probability Mass Function (PMF) values:");
    println!("  k    P(X = k)");
    println!("---------------");
    for k in 0..15 {
        let k_f = k as f64;
        let pmf = nb.pmf(k_f);
        println!("  {:<3}  {:.6}", k, pmf);
    }
    println!();

    // Calculate and display CDF values
    println!("Cumulative Distribution Function (CDF) values:");
    println!("  k    P(X ≤ k)");
    println!("---------------");
    for k in 0..15 {
        let k_f = k as f64;
        let cdf = nb.cdf(k_f);
        println!("  {:<3}  {:.6}", k, cdf);
    }
    println!();

    // Calculate quantiles
    println!("Quantiles (Inverse CDF):");
    println!("  p      k where P(X ≤ k) ≥ p");
    println!("------------------------------");
    for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].iter() {
        let quant = nb.ppf(*p)?;
        println!("  {:.2}    {}", p, quant);
    }
    println!();

    // Calculate and display statistical properties
    println!("Statistical Properties:");
    println!("- Mean:            {:.6}", nb.mean());
    println!("- Variance:        {:.6}", nb.var());
    println!("- Std Deviation:   {:.6}", nb.std());
    println!("- Skewness:        {:.6}", nb.skewness());
    println!("- Excess Kurtosis: {:.6}", nb.kurtosis());
    println!("- Mode:            {}", nb.mode());
    println!("- Entropy:         {:.6}", nb.entropy());
    println!();

    // Generate random samples
    println!("Random Samples:");
    let samples = nb.rvs(10)?;
    for (i, sample) in samples.iter().enumerate() {
        print!("{:.0}", sample);
        if i < samples.len() - 1 {
            print!(", ");
        }
    }
    println!("\n");

    // Educational examples using the Negative Binomial distribution
    println!("Real-world Applications:");
    println!("1. Modeling the number of failures before achieving a fixed number of successes");
    println!("2. Insurance: Number of accidents before r claims are processed");
    println!("3. Quality control: Number of defective items before finding r good ones");
    println!("4. Epidemiology: Modeling disease spread and contagion processes");
    println!("5. Ecology: Modeling species abundance or spatial clustering of organisms");
    println!("6. Marketing: Number of contacts before achieving r sales");
    println!();

    // Comparison with related distributions
    println!("Related Distributions:");
    println!("- Geometric distribution is a special case where r = 1");
    let geom = nbinom(1.0f64, 0.3)?;
    println!("  - Geometric(0.3) mean: {:.6}", geom.mean());
    println!("  - NegativeBinomial(1, 0.3) mean: {:.6}", geom.mean());
    println!("- Binomial distribution models number of successes in n fixed trials");
    println!("- Poisson distribution can be derived as a limiting case");
    println!();

    Ok(())
}
