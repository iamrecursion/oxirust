use scirs2_stats::distributions::logistic::Logistic;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Logistic Distribution Example");
    println!("----------------------------");

    // Create Logistic distributions with different parameters
    let standard_logistic = Logistic::new(0.0, 1.0)?; // Standard Logistic (loc=0, scale=1)
    let shifted_logistic = Logistic::new(2.0, 1.0)?; // Shifted to location=2
    let narrow_logistic = Logistic::new(0.0, 0.5)?; // Narrower with scale=0.5
                                                    // We'll define but not use this custom distribution to demonstrate other parameter values
    let _custom_logistic = Logistic::new(-1.0, 2.0)?; // Custom with loc=-1, scale=2

    // Print parameters and properties
    println!("\n1. Standard Logistic (loc=0, scale=1)");
    println!("   Location parameter: {}", standard_logistic.loc);
    println!("   Scale parameter: {}", standard_logistic.scale);
    println!("   Mean: {}", standard_logistic.mean());
    println!("   Variance: {:.6}", standard_logistic.var());
    println!("   Standard deviation: {:.6}", standard_logistic.std());
    println!("   Median: {}", standard_logistic.median());
    println!("   Mode: {}", standard_logistic.mode());
    println!("   Skewness: {}", standard_logistic.skewness());
    println!("   Kurtosis: {}", standard_logistic.kurtosis());
    println!("   Entropy: {:.6}", standard_logistic.entropy());
    println!(
        "   Interquartile range (IQR): {:.6}",
        standard_logistic.iqr()
    );

    println!("\n2. Shifted Logistic (loc=2, scale=1)");
    println!("   Location parameter: {}", shifted_logistic.loc);
    println!("   Scale parameter: {}", shifted_logistic.scale);
    println!("   Mean: {}", shifted_logistic.mean());
    println!("   Variance: {:.6}", shifted_logistic.var());
    println!("   Standard deviation: {:.6}", shifted_logistic.std());
    println!("   Median: {}", shifted_logistic.median());
    println!("   Mode: {}", shifted_logistic.mode());
    println!("   Entropy: {:.6}", shifted_logistic.entropy());

    println!("\n3. Narrow Logistic (loc=0, scale=0.5)");
    println!("   Location parameter: {}", narrow_logistic.loc);
    println!("   Scale parameter: {}", narrow_logistic.scale);
    println!("   Mean: {}", narrow_logistic.mean());
    println!("   Variance: {:.6}", narrow_logistic.var());
    println!("   Standard deviation: {:.6}", narrow_logistic.std());
    println!("   Median: {}", narrow_logistic.median());
    println!("   Mode: {}", narrow_logistic.mode());
    println!("   Entropy: {:.6}", narrow_logistic.entropy());
    println!("   Interquartile range (IQR): {:.6}", narrow_logistic.iqr());

    // Calculate PDF values at different points
    println!("\nPDF Values:");
    println!("                      x=-3      x=-1      x=0       x=1       x=3");
    println!(
        "Standard Logistic:   {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_logistic.pdf(-3.0),
        standard_logistic.pdf(-1.0),
        standard_logistic.pdf(0.0),
        standard_logistic.pdf(1.0),
        standard_logistic.pdf(3.0)
    );
    println!(
        "Shifted Logistic:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_logistic.pdf(-3.0),
        shifted_logistic.pdf(-1.0),
        shifted_logistic.pdf(0.0),
        shifted_logistic.pdf(1.0),
        shifted_logistic.pdf(3.0)
    );
    println!(
        "Narrow Logistic:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_logistic.pdf(-3.0),
        narrow_logistic.pdf(-1.0),
        narrow_logistic.pdf(0.0),
        narrow_logistic.pdf(1.0),
        narrow_logistic.pdf(3.0)
    );

    // Calculate CDF values at different points
    println!("\nCDF Values:");
    println!("                      x=-3      x=-1      x=0       x=1       x=3");
    println!(
        "Standard Logistic:   {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_logistic.cdf(-3.0),
        standard_logistic.cdf(-1.0),
        standard_logistic.cdf(0.0),
        standard_logistic.cdf(1.0),
        standard_logistic.cdf(3.0)
    );
    println!(
        "Shifted Logistic:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_logistic.cdf(-3.0),
        shifted_logistic.cdf(-1.0),
        shifted_logistic.cdf(0.0),
        shifted_logistic.cdf(1.0),
        shifted_logistic.cdf(3.0)
    );
    println!(
        "Narrow Logistic:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_logistic.cdf(-3.0),
        narrow_logistic.cdf(-1.0),
        narrow_logistic.cdf(0.0),
        narrow_logistic.cdf(1.0),
        narrow_logistic.cdf(3.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                      p=0.1     p=0.25    p=0.5     p=0.75    p=0.9");
    println!(
        "Standard Logistic:   {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_logistic.ppf(0.1)?,
        standard_logistic.ppf(0.25)?,
        standard_logistic.ppf(0.5)?,
        standard_logistic.ppf(0.75)?,
        standard_logistic.ppf(0.9)?
    );
    println!(
        "Shifted Logistic:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_logistic.ppf(0.1)?,
        shifted_logistic.ppf(0.25)?,
        shifted_logistic.ppf(0.5)?,
        shifted_logistic.ppf(0.75)?,
        shifted_logistic.ppf(0.9)?
    );
    println!(
        "Narrow Logistic:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_logistic.ppf(0.1)?,
        narrow_logistic.ppf(0.25)?,
        narrow_logistic.ppf(0.5)?,
        narrow_logistic.ppf(0.75)?,
        narrow_logistic.ppf(0.9)?
    );

    // Generate and display random samples
    println!("\nRandom Samples from Standard Logistic:");
    let samples = standard_logistic.rvs(10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Check the closeness of inverse CDF and CDF operations
    println!("\nVerifying the inverse relationship between CDF and PPF:");

    let test_values = [-2.0, -0.5, 0.0, 0.5, 2.0];
    for &x in &test_values {
        let p = standard_logistic.cdf(x);
        let x_back = standard_logistic.ppf(p)?;
        let diff = f64::abs(x - x_back);
        println!(
            "  x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p, x_back, diff
        );
    }

    println!("\nLogistic Distribution Applications:");
    println!("1. Logistic regression for binary classification");
    println!("2. Neural networks (sigmoid activation function)");
    println!("3. Population growth models");
    println!("4. Item response theory in psychometrics");
    println!("5. Modeling technological adoption and diffusion");
    println!("6. Survival analysis and reliability engineering");

    println!("\nUnique Properties of the Logistic Distribution:");
    println!("1. Similar shape to normal but with heavier tails");
    println!("2. Symmetrical around its mean/median/mode");
    println!("3. CDF has a simple closed-form expression (unlike normal)");
    println!("4. Kurtosis of 1.2 (moderately heavy-tailed)");
    println!("5. Variance is π²/3 × scale² ≈ 3.29 × scale²");
    println!("6. Used as a smooth approximation of the step function");

    println!("\nKey formulas:");
    println!("- PDF: f(x) = e^(-z) / (scale × (1 + e^(-z))²) where z = (x-loc)/scale");
    println!("- CDF: F(x) = 1 / (1 + e^(-z))");
    println!("- Quantile: Q(p) = loc + scale × ln(p/(1-p))");
    println!("- Mean/Median/Mode = loc");
    println!("- Variance = π²/3 × scale²");
    println!("- Standard deviation = π/√3 × scale ≈ 1.81 × scale");
    println!("- Entropy = 2 + ln(scale)");
    println!("- IQR = scale × ln(3) ≈ 1.1 × scale");

    Ok(())
}
