use scirs2_stats::distributions::lognormal::Lognormal;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Lognormal Distribution Example");
    println!("-----------------------------");

    // Create a standard lognormal distribution (mu=0, sigma=1, loc=0)
    let standard_lognorm = Lognormal::new(0.0, 1.0, 0.0)?;

    println!("Standard Lognormal Distribution (mu=0, sigma=1, loc=0)");
    println!("Parameters:");
    println!("  mu (mean of log(X)): {}", standard_lognorm.mu);
    println!("  sigma (std dev of log(X)): {}", standard_lognorm.sigma);
    println!("  loc (location parameter): {}", standard_lognorm.loc);
    println!();

    // Calculate distribution properties
    println!("Distribution properties:");
    println!("  Mean: {:.7}", standard_lognorm.mean());
    println!("  Variance: {:.7}", standard_lognorm.var());
    println!("  Median: {:.7}", standard_lognorm.median());
    println!("  Mode: {:.7}", standard_lognorm.mode());
    println!();

    // Calculate PDF at various points
    println!("PDF values at different points:");
    println!("  PDF at x = 0.5: {:.7}", standard_lognorm.pdf(0.5));
    println!("  PDF at x = 1.0: {:.7}", standard_lognorm.pdf(1.0));
    println!("  PDF at x = 2.0: {:.7}", standard_lognorm.pdf(2.0));
    println!("  PDF at x = 0.0: {:.7}", standard_lognorm.pdf(0.0)); // Should be 0
    println!("  PDF at x = -1.0: {:.7}", standard_lognorm.pdf(-1.0)); // Should be 0
    println!();

    // Calculate CDF at various points
    println!("CDF values at different points:");
    println!("  CDF at x = 0.5: {:.7}", standard_lognorm.cdf(0.5));
    println!("  CDF at x = 1.0: {:.7}", standard_lognorm.cdf(1.0));
    println!("  CDF at x = 2.0: {:.7}", standard_lognorm.cdf(2.0));
    println!("  CDF at x = 0.0: {:.7}", standard_lognorm.cdf(0.0)); // Should be 0
    println!("  CDF at x = -1.0: {:.7}", standard_lognorm.cdf(-1.0)); // Should be 0
    println!();

    // Calculate quantiles (inverse CDF)
    println!("Quantiles:");
    println!("  10th percentile: {:.7}", standard_lognorm.ppf(0.1)?);
    println!("  25th percentile: {:.7}", standard_lognorm.ppf(0.25)?);
    println!("  50th percentile: {:.7}", standard_lognorm.ppf(0.5)?);
    println!("  75th percentile: {:.7}", standard_lognorm.ppf(0.75)?);
    println!("  90th percentile: {:.7}", standard_lognorm.ppf(0.9)?);
    println!();

    // Generate random samples
    println!("Generating 10 random samples:");
    let samples = standard_lognorm.rvs(10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }
    println!();

    // Create a custom lognormal distribution (mu=1, sigma=0.5, loc=2)
    let custom_lognorm = Lognormal::new(1.0, 0.5, 2.0)?;

    println!("Custom Lognormal Distribution (mu=1, sigma=0.5, loc=2)");
    println!("Parameters:");
    println!("  mu (mean of log(X)): {}", custom_lognorm.mu);
    println!("  sigma (std dev of log(X)): {}", custom_lognorm.sigma);
    println!("  loc (location parameter): {}", custom_lognorm.loc);
    println!();

    // Calculate distribution properties
    println!("Distribution properties:");
    println!("  Mean: {:.7}", custom_lognorm.mean());
    println!("  Variance: {:.7}", custom_lognorm.var());
    println!("  Median: {:.7}", custom_lognorm.median());
    println!("  Mode: {:.7}", custom_lognorm.mode());
    println!();

    // Comparing with the normal distribution
    println!("Relationship with the normal distribution:");
    println!("  If X ~ Lognormal(μ, σ², loc), then ln(X - loc) ~ Normal(μ, σ²)");
    println!("  Mean = loc + exp(μ + σ²/2)");
    println!("  Median = loc + exp(μ)");
    println!("  Mode = loc + exp(μ - σ²)");
    println!("  Variance = exp(2μ + σ²) * (exp(σ²) - 1)");
    println!();

    // Applications
    println!("Applications of the Lognormal distribution:");
    println!("1. Modeling sizes of biological organisms");
    println!("2. Financial modeling (stock prices, income distributions)");
    println!("3. Failure times in reliability analysis");
    println!("4. Environmental science (pollutant concentrations)");
    println!("5. Network traffic analysis");

    Ok(())
}
