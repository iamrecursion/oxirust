use scirs2_stats::distributions::bernoulli::Bernoulli;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Bernoulli Distribution Example");
    println!("-----------------------------");

    // Create Bernoulli distributions with different parameters
    let bern_low = Bernoulli::new(0.3)?; // Low success probability
    let bern_fair = Bernoulli::new(0.5)?; // Fair coin (p = 0.5)
    let bern_high = Bernoulli::new(0.8)?; // High success probability

    // Print parameters and properties
    println!("\n1. Low Success Probability (p=0.3)");
    println!("   Success probability: {}", bern_low.p);
    println!("   Mean: {}", bern_low.mean());
    println!("   Variance: {}", bern_low.var());
    println!("   Standard deviation: {:.6}", bern_low.std());
    println!("   Median: {}", bern_low.median());
    println!("   Mode: {}", bern_low.mode());
    println!("   Skewness: {:.6}", bern_low.skewness());
    println!("   Kurtosis: {:.6}", bern_low.kurtosis());
    println!("   Entropy: {:.6}", bern_low.entropy());

    println!("\n2. Fair Bernoulli (p=0.5)");
    println!("   Success probability: {}", bern_fair.p);
    println!("   Mean: {}", bern_fair.mean());
    println!("   Variance: {}", bern_fair.var());
    println!("   Standard deviation: {:.6}", bern_fair.std());
    println!("   Median: {}", bern_fair.median());
    println!("   Mode: {}", bern_fair.mode());
    println!("   Skewness: {:.6}", bern_fair.skewness());
    println!("   Kurtosis: {:.6}", bern_fair.kurtosis());
    println!("   Entropy: {:.6}", bern_fair.entropy());

    println!("\n3. High Success Probability (p=0.8)");
    println!("   Success probability: {}", bern_high.p);
    println!("   Mean: {}", bern_high.mean());
    println!("   Variance: {}", bern_high.var());
    println!("   Standard deviation: {:.6}", bern_high.std());
    println!("   Median: {}", bern_high.median());
    println!("   Mode: {}", bern_high.mode());
    println!("   Skewness: {:.6}", bern_high.skewness());
    println!("   Kurtosis: {:.6}", bern_high.kurtosis());
    println!("   Entropy: {:.6}", bern_high.entropy());

    // Calculate PMF values
    println!("\nPMF Values:");
    println!("                     k=0      k=1      k=2");
    println!(
        "Low (p=0.3):        {:.6} {:.6} {:.6}",
        bern_low.pmf(0.0),
        bern_low.pmf(1.0),
        bern_low.pmf(2.0)
    );
    println!(
        "Fair (p=0.5):       {:.6} {:.6} {:.6}",
        bern_fair.pmf(0.0),
        bern_fair.pmf(1.0),
        bern_fair.pmf(2.0)
    );
    println!(
        "High (p=0.8):       {:.6} {:.6} {:.6}",
        bern_high.pmf(0.0),
        bern_high.pmf(1.0),
        bern_high.pmf(2.0)
    );

    // Calculate CDF values
    println!("\nCDF Values:");
    println!("                     k=-1     k=0      k=0.5    k=1      k=2");
    println!(
        "Low (p=0.3):        {:.6} {:.6} {:.6} {:.6} {:.6}",
        bern_low.cdf(-1.0),
        bern_low.cdf(0.0),
        bern_low.cdf(0.5),
        bern_low.cdf(1.0),
        bern_low.cdf(2.0)
    );
    println!(
        "Fair (p=0.5):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        bern_fair.cdf(-1.0),
        bern_fair.cdf(0.0),
        bern_fair.cdf(0.5),
        bern_fair.cdf(1.0),
        bern_fair.cdf(2.0)
    );
    println!(
        "High (p=0.8):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        bern_high.cdf(-1.0),
        bern_high.cdf(0.0),
        bern_high.cdf(0.5),
        bern_high.cdf(1.0),
        bern_high.cdf(2.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                     p=0      p=0.3    p=0.5    p=0.7    p=1");
    println!(
        "Low (p=0.3):        {:.6} {:.6} {:.6} {:.6} {:.6}",
        bern_low.ppf(0.0)?,
        bern_low.ppf(0.3)?,
        bern_low.ppf(0.5)?,
        bern_low.ppf(0.7)?,
        bern_low.ppf(1.0)?
    );
    println!(
        "Fair (p=0.5):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        bern_fair.ppf(0.0)?,
        bern_fair.ppf(0.3)?,
        bern_fair.ppf(0.5)?,
        bern_fair.ppf(0.7)?,
        bern_fair.ppf(1.0)?
    );
    println!(
        "High (p=0.8):       {:.6} {:.6} {:.6} {:.6} {:.6}",
        bern_high.ppf(0.0)?,
        bern_high.ppf(0.3)?,
        bern_high.ppf(0.5)?,
        bern_high.ppf(0.7)?,
        bern_high.ppf(1.0)?
    );

    // Generate and display random samples
    println!("\nRandom Samples from Fair Bernoulli (p=0.5):");
    let samples = bern_fair.rvs(20)?;
    for (i, &sample) in samples.iter().enumerate() {
        if i % 5 == 0 && i > 0 {
            println!();
        }
        print!("  Sample {}: {}   ", i + 1, sample as i32);
    }
    println!("\n");

    // Count the number of successes (1s) in the sample
    let success_count = samples.iter().filter(|&&x| x == 1.0).count();
    println!(
        "Successes: {} out of {} trials ({}%)",
        success_count,
        samples.len(),
        100.0 * success_count as f64 / samples.len() as f64
    );

    println!("\nBernoulli Distribution Applications:");
    println!("1. Modeling coin flips, dice rolls, or any binary outcome");
    println!("2. Binary classification problems");
    println!("3. Success/failure of a single trial");
    println!("4. Quality control (pass/fail)");
    println!("5. Modeling binary states (on/off, yes/no, true/false)");
    println!("6. Basis for more complex distributions like Binomial");

    println!("\nUnique Properties of the Bernoulli Distribution:");
    println!("1. Simplest discrete probability distribution");
    println!("2. Only takes values 0 and 1");
    println!("3. Variance is maximized when p = 0.5");
    println!("4. Skewness = 0 when p = 0.5 (symmetric)");
    println!("5. Special case of binomial distribution with n = 1");
    println!("6. Variance is always less than the mean");

    println!("\nKey formulas:");
    println!("- PMF: P(X = k) = p^k * (1-p)^(1-k) for k ∈ {{0, 1}}");
    println!("- CDF: F(x) = 0 for x < 0");
    println!("       F(x) = 1-p for 0 ≤ x < 1");
    println!("       F(x) = 1 for x ≥ 1");
    println!("- Mean = p");
    println!("- Variance = p(1-p)");
    println!("- Skewness = (1-2p) / √(p(1-p))");
    println!("- Kurtosis = (1-6p(1-p)) / (p(1-p))");
    println!("- Entropy = -p·ln(p) - (1-p)·ln(1-p)");

    Ok(())
}
