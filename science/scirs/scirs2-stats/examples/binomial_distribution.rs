use scirs2_stats::distributions::binomial::Binomial;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Binomial Distribution Example");
    println!("----------------------------");

    // Create Binomial distributions with different parameters
    let binom_small = Binomial::new(5, 0.4)?; // Small number of trials
    let binom_med = Binomial::new(10, 0.5)?; // Medium symmetric case
    let binom_large = Binomial::new(20, 0.25)?; // Large asymmetric case

    // Print parameters and properties
    println!("\n1. Small Binomial (n=5, p=0.4)");
    println!("   Number of trials: {}", binom_small.n);
    println!("   Success probability: {}", binom_small.p);
    println!("   Mean: {}", binom_small.mean());
    println!("   Variance: {}", binom_small.var());
    println!("   Standard deviation: {:.6}", binom_small.std());
    println!("   Median: {}", binom_small.median());
    println!("   Mode(s): {:?}", binom_small.mode());
    println!("   Skewness: {:.6}", binom_small.skewness());
    println!("   Kurtosis: {:.6}", binom_small.kurtosis());
    println!("   Entropy (approximation): {:.6}", binom_small.entropy());

    println!("\n2. Medium Symmetric Binomial (n=10, p=0.5)");
    println!("   Number of trials: {}", binom_med.n);
    println!("   Success probability: {}", binom_med.p);
    println!("   Mean: {}", binom_med.mean());
    println!("   Variance: {}", binom_med.var());
    println!("   Standard deviation: {:.6}", binom_med.std());
    println!("   Median: {}", binom_med.median());
    println!("   Mode(s): {:?}", binom_med.mode());
    println!("   Skewness: {:.6}", binom_med.skewness());
    println!("   Kurtosis: {:.6}", binom_med.kurtosis());
    println!("   Entropy (approximation): {:.6}", binom_med.entropy());

    println!("\n3. Large Asymmetric Binomial (n=20, p=0.25)");
    println!("   Number of trials: {}", binom_large.n);
    println!("   Success probability: {}", binom_large.p);
    println!("   Mean: {}", binom_large.mean());
    println!("   Variance: {}", binom_large.var());
    println!("   Standard deviation: {:.6}", binom_large.std());
    println!("   Median: {}", binom_large.median());
    println!("   Mode(s): {:?}", binom_large.mode());
    println!("   Skewness: {:.6}", binom_large.skewness());
    println!("   Kurtosis: {:.6}", binom_large.kurtosis());
    println!("   Entropy (approximation): {:.6}", binom_large.entropy());

    // Calculate PMF values
    println!("\nPMF Values:");
    println!("                    k=0      k=2      k=5      k=8      k=10");
    println!(
        "Small (n=5, p=0.4): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_small.pmf(0.0),
        binom_small.pmf(2.0),
        binom_small.pmf(5.0),
        binom_small.pmf(8.0),
        binom_small.pmf(10.0)
    );
    println!(
        "Medium (n=10, p=0.5): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_med.pmf(0.0),
        binom_med.pmf(2.0),
        binom_med.pmf(5.0),
        binom_med.pmf(8.0),
        binom_med.pmf(10.0)
    );
    println!(
        "Large (n=20, p=0.25): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_large.pmf(0.0),
        binom_large.pmf(2.0),
        binom_large.pmf(5.0),
        binom_large.pmf(8.0),
        binom_large.pmf(10.0)
    );

    // Calculate log-PMF values (useful for avoiding underflow with large n)
    println!("\nLog-PMF Values:");
    println!("                    k=0      k=2      k=5      k=8      k=10");
    println!(
        "Small (n=5, p=0.4): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_small.log_pmf(0.0),
        binom_small.log_pmf(2.0),
        binom_small.log_pmf(5.0),
        binom_small.log_pmf(8.0),
        binom_small.log_pmf(10.0)
    );
    println!(
        "Medium (n=10, p=0.5): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_med.log_pmf(0.0),
        binom_med.log_pmf(2.0),
        binom_med.log_pmf(5.0),
        binom_med.log_pmf(8.0),
        binom_med.log_pmf(10.0)
    );
    println!(
        "Large (n=20, p=0.25): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_large.log_pmf(0.0),
        binom_large.log_pmf(2.0),
        binom_large.log_pmf(5.0),
        binom_large.log_pmf(8.0),
        binom_large.log_pmf(10.0)
    );

    // Calculate CDF values
    println!("\nCDF Values:");
    println!("                    k=0      k=2      k=5      k=8      k=10");
    println!(
        "Small (n=5, p=0.4): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_small.cdf(0.0),
        binom_small.cdf(2.0),
        binom_small.cdf(5.0),
        binom_small.cdf(8.0),
        binom_small.cdf(10.0)
    );
    println!(
        "Medium (n=10, p=0.5): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_med.cdf(0.0),
        binom_med.cdf(2.0),
        binom_med.cdf(5.0),
        binom_med.cdf(8.0),
        binom_med.cdf(10.0)
    );
    println!(
        "Large (n=20, p=0.25): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_large.cdf(0.0),
        binom_large.cdf(2.0),
        binom_large.cdf(5.0),
        binom_large.cdf(8.0),
        binom_large.cdf(10.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                    p=0      p=0.25   p=0.5    p=0.75   p=1");
    println!(
        "Small (n=5, p=0.4): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_small.ppf(0.0)?,
        binom_small.ppf(0.25)?,
        binom_small.ppf(0.5)?,
        binom_small.ppf(0.75)?,
        binom_small.ppf(1.0)?
    );
    println!(
        "Medium (n=10, p=0.5): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_med.ppf(0.0)?,
        binom_med.ppf(0.25)?,
        binom_med.ppf(0.5)?,
        binom_med.ppf(0.75)?,
        binom_med.ppf(1.0)?
    );
    println!(
        "Large (n=20, p=0.25): {:.6} {:.6} {:.6} {:.6} {:.6}",
        binom_large.ppf(0.0)?,
        binom_large.ppf(0.25)?,
        binom_large.ppf(0.5)?,
        binom_large.ppf(0.75)?,
        binom_large.ppf(1.0)?
    );

    // Generate and display random samples
    println!("\nRandom Samples from Medium Binomial (n=10, p=0.5):");
    let samples = binom_med.rvs(20)?;
    for (i, &sample) in samples.iter().enumerate() {
        if i % 5 == 0 && i > 0 {
            println!();
        }
        print!("  Sample {}: {}   ", i + 1, sample as i32);
    }
    println!("\n");

    // Calculate summary statistics of the sample
    let sum: f64 = samples.iter().sum();
    let mean = sum / samples.len() as f64;
    let count_5_or_more = samples.iter().filter(|&&x| x >= 5.0).count();
    println!("Sample mean: {:.2}", mean);
    println!(
        "Number of trials with 5 or more successes: {} ({}%)",
        count_5_or_more,
        100.0 * count_5_or_more as f64 / samples.len() as f64
    );

    println!("\nBinomial Distribution Applications:");
    println!("1. Number of successes in a fixed number of independent trials");
    println!("2. Quality control (number of defects in a batch)");
    println!("3. Epidemiology (number of disease cases in a population)");
    println!("4. Finance (number of stocks that increase in value)");
    println!("5. Reliability testing (number of components that fail)");
    println!("6. A/B testing (number of conversions in marketing tests)");

    println!("\nUnique Properties of the Binomial Distribution:");
    println!("1. Sum of n independent Bernoulli random variables with same p");
    println!("2. Approaches normal distribution as n increases (by CLT)");
    println!("3. Symmetric when p = 0.5, skewed otherwise");
    println!("4. Mean (n*p) equals the variance (n*p*(1-p)) when p = 1/(n+1)");
    println!("5. Special cases: Bernoulli (n=1) and degenerate (p=0 or p=1)");
    println!("6. Each trial has only two possible outcomes (success/failure)");

    println!("\nKey formulas:");
    println!("- PMF: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)");
    println!("- CDF: F(x) = Sum from k=0 to floor(x) of [C(n,k) * p^k * (1-p)^(n-k)]");
    println!("- Mean = n*p");
    println!("- Variance = n*p*(1-p)");
    println!("- Skewness = (1-2p) / sqrt(n*p*(1-p))");
    println!("- Kurtosis = (1-6p(1-p)) / (n*p*(1-p))");
    println!("- Mode = floor((n+1)p) or ceil((n+1)p)-1 or both");

    Ok(())
}
