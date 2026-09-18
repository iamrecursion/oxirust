use scirs2_stats::distributions::geometric::Geometric;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Geometric Distribution Example");
    println!("-----------------------------");

    // Create Geometric distributions with different parameters
    let geom_low = Geometric::new(0.2)?; // Low success probability (long wait)
    let geom_med = Geometric::new(0.5)?; // Medium success probability
    let geom_high = Geometric::new(0.8)?; // High success probability (short wait)

    // Print parameters and properties
    println!("\n1. Low Success Probability (p=0.2)");
    println!("   Success probability: {}", geom_low.p);
    println!("   Mean: {}", geom_low.mean());
    println!("   Variance: {}", geom_low.var());
    println!("   Standard deviation: {:.6}", geom_low.std());
    println!("   Median: {}", geom_low.median());
    println!("   Mode: {}", geom_low.mode());
    println!("   Skewness: {:.6}", geom_low.skewness());
    println!("   Kurtosis: {:.6}", geom_low.kurtosis());
    println!("   Entropy: {:.6}", geom_low.entropy());

    println!("\n2. Medium Success Probability (p=0.5)");
    println!("   Success probability: {}", geom_med.p);
    println!("   Mean: {}", geom_med.mean());
    println!("   Variance: {}", geom_med.var());
    println!("   Standard deviation: {:.6}", geom_med.std());
    println!("   Median: {}", geom_med.median());
    println!("   Mode: {}", geom_med.mode());
    println!("   Skewness: {:.6}", geom_med.skewness());
    println!("   Kurtosis: {:.6}", geom_med.kurtosis());
    println!("   Entropy: {:.6}", geom_med.entropy());

    println!("\n3. High Success Probability (p=0.8)");
    println!("   Success probability: {}", geom_high.p);
    println!("   Mean: {}", geom_high.mean());
    println!("   Variance: {}", geom_high.var());
    println!("   Standard deviation: {:.6}", geom_high.std());
    println!("   Median: {}", geom_high.median());
    println!("   Mode: {}", geom_high.mode());
    println!("   Skewness: {:.6}", geom_high.skewness());
    println!("   Kurtosis: {:.6}", geom_high.kurtosis());
    println!("   Entropy: {:.6}", geom_high.entropy());

    // Calculate PMF values
    println!("\nPMF Values (probability of exactly k failures before first success):");
    println!("                    k=0      k=1      k=2      k=5      k=10");
    println!(
        "Low (p=0.2):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_low.pmf(0.0),
        geom_low.pmf(1.0),
        geom_low.pmf(2.0),
        geom_low.pmf(5.0),
        geom_low.pmf(10.0)
    );
    println!(
        "Medium (p=0.5):    {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_med.pmf(0.0),
        geom_med.pmf(1.0),
        geom_med.pmf(2.0),
        geom_med.pmf(5.0),
        geom_med.pmf(10.0)
    );
    println!(
        "High (p=0.8):      {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_high.pmf(0.0),
        geom_high.pmf(1.0),
        geom_high.pmf(2.0),
        geom_high.pmf(5.0),
        geom_high.pmf(10.0)
    );

    // Calculate CDF values
    println!("\nCDF Values (probability of at most k failures before first success):");
    println!("                    k=0      k=1      k=2      k=5      k=10");
    println!(
        "Low (p=0.2):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_low.cdf(0.0),
        geom_low.cdf(1.0),
        geom_low.cdf(2.0),
        geom_low.cdf(5.0),
        geom_low.cdf(10.0)
    );
    println!(
        "Medium (p=0.5):    {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_med.cdf(0.0),
        geom_med.cdf(1.0),
        geom_med.cdf(2.0),
        geom_med.cdf(5.0),
        geom_med.cdf(10.0)
    );
    println!(
        "High (p=0.8):      {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_high.cdf(0.0),
        geom_high.cdf(1.0),
        geom_high.cdf(2.0),
        geom_high.cdf(5.0),
        geom_high.cdf(10.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                    p=0.1    p=0.25   p=0.5    p=0.75   p=0.9");
    println!(
        "Low (p=0.2):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_low.ppf(0.1)?,
        geom_low.ppf(0.25)?,
        geom_low.ppf(0.5)?,
        geom_low.ppf(0.75)?,
        geom_low.ppf(0.9)?
    );
    println!(
        "Medium (p=0.5):    {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_med.ppf(0.1)?,
        geom_med.ppf(0.25)?,
        geom_med.ppf(0.5)?,
        geom_med.ppf(0.75)?,
        geom_med.ppf(0.9)?
    );
    println!(
        "High (p=0.8):      {:.6} {:.6} {:.6} {:.6} {:.6}",
        geom_high.ppf(0.1)?,
        geom_high.ppf(0.25)?,
        geom_high.ppf(0.5)?,
        geom_high.ppf(0.75)?,
        geom_high.ppf(0.9)?
    );

    // Generate and display random samples
    println!("\nRandom Samples from Medium Geometric (p=0.5):");
    let samples = geom_med.rvs(20)?;
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
    let count_large = samples.iter().filter(|&&x| x >= 3.0).count();
    println!("Sample mean: {:.2}", mean);
    println!(
        "Number of trials with 3 or more failures: {} ({}%)",
        count_large,
        100.0 * count_large as f64 / samples.len() as f64
    );

    println!("\nGeometric Distribution Applications:");
    println!("1. Number of failures before first success in independent trials");
    println!("2. Number of attempts until a rare event occurs");
    println!("3. Waiting time models (discrete time)");
    println!("4. Reliability testing (number of trials until first defect)");
    println!("5. Modeling retry attempts in computer systems");
    println!("6. Sequential sampling schemes in quality control");

    println!("\nUnique Properties of the Geometric Distribution:");
    println!("1. The only discrete distribution with the 'memoryless' property");
    println!("2. Discrete analogue of the exponential distribution");
    println!("3. Special case of negative binomial distribution with r=1");
    println!("4. Always has mode at 0 (most likely outcome is immediate success)");
    println!("5. Mean (1-p)/p always exceeds the median");
    println!("6. Has the highest entropy among all distributions on non-negative integers with given mean");

    println!("\nKey formulas:");
    println!("- PMF: P(X = k) = p * (1-p)^k");
    println!("- CDF: F(k) = 1 - (1-p)^(k+1)");
    println!("- Mean = (1-p)/p");
    println!("- Variance = (1-p)/p^2");
    println!("- Skewness = (2-p)/sqrt(1-p)");
    println!("- Kurtosis = 6 + p^2/(1-p)");
    println!("- Mode = 0 (always)");
    println!("- Median = ceiling(ln(0.5)/ln(1-p)) - 1");
    println!("- Entropy = -[(1-p)*ln(1-p) + p*ln(p)]/p");

    Ok(())
}
