use scirs2_stats::distributions::laplace::Laplace;
use scirs2_stats::traits::{ContinuousDistribution, Distribution};
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Laplace (Double Exponential) Distribution Example");
    println!("------------------------------------------------");

    // Create Laplace distributions with different parameters
    let standard_laplace = Laplace::new(0.0, 1.0)?; // Standard Laplace (loc=0, scale=1)
    let shifted_laplace = Laplace::new(2.0, 1.0)?; // Shifted to location=2
    let narrow_laplace = Laplace::new(0.0, 0.5)?; // Narrower with scale=0.5
                                                  // We'll define but not use this custom distribution to demonstrate other parameter values
    let _custom_laplace = Laplace::new(-1.0, 2.0)?; // Custom with loc=-1, scale=2

    // Print parameters and properties
    println!("\n1. Standard Laplace (loc=0, scale=1)");
    println!("   Location parameter: {}", standard_laplace.loc);
    println!("   Scale parameter: {}", standard_laplace.scale);
    println!("   Mean: {}", standard_laplace.mean());
    println!("   Variance: {}", standard_laplace.var());
    println!("   Standard deviation: {:.6}", standard_laplace.std());
    println!("   Median: {}", standard_laplace.median());
    println!("   Mode: {}", standard_laplace.mode());
    println!("   Skewness: {}", standard_laplace.skewness());
    println!("   Kurtosis: {}", standard_laplace.kurtosis());
    println!("   Entropy: {:.6}", standard_laplace.entropy());

    println!("\n2. Shifted Laplace (loc=2, scale=1)");
    println!("   Location parameter: {}", shifted_laplace.loc);
    println!("   Scale parameter: {}", shifted_laplace.scale);
    println!("   Mean: {}", shifted_laplace.mean());
    println!("   Variance: {}", shifted_laplace.var());
    println!("   Standard deviation: {:.6}", shifted_laplace.std());
    println!("   Median: {}", shifted_laplace.median());
    println!("   Mode: {}", shifted_laplace.mode());
    println!("   Entropy: {:.6}", shifted_laplace.entropy());

    println!("\n3. Narrow Laplace (loc=0, scale=0.5)");
    println!("   Location parameter: {}", narrow_laplace.loc);
    println!("   Scale parameter: {}", narrow_laplace.scale);
    println!("   Mean: {}", narrow_laplace.mean());
    println!("   Variance: {}", narrow_laplace.var());
    println!("   Standard deviation: {:.6}", narrow_laplace.std());
    println!("   Median: {}", narrow_laplace.median());
    println!("   Mode: {}", narrow_laplace.mode());
    println!("   Entropy: {:.6}", narrow_laplace.entropy());

    // Calculate PDF values at different points
    println!("\nPDF Values:");
    println!("                      x=-3      x=-1      x=0       x=1       x=3");
    println!(
        "Standard Laplace:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_laplace.pdf(-3.0),
        standard_laplace.pdf(-1.0),
        standard_laplace.pdf(0.0),
        standard_laplace.pdf(1.0),
        standard_laplace.pdf(3.0)
    );
    println!(
        "Shifted Laplace:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_laplace.pdf(-3.0),
        shifted_laplace.pdf(-1.0),
        shifted_laplace.pdf(0.0),
        shifted_laplace.pdf(1.0),
        shifted_laplace.pdf(3.0)
    );
    println!(
        "Narrow Laplace:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_laplace.pdf(-3.0),
        narrow_laplace.pdf(-1.0),
        narrow_laplace.pdf(0.0),
        narrow_laplace.pdf(1.0),
        narrow_laplace.pdf(3.0)
    );

    // Calculate CDF values at different points
    println!("\nCDF Values:");
    println!("                      x=-3      x=-1      x=0       x=1       x=3");
    println!(
        "Standard Laplace:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_laplace.cdf(-3.0),
        standard_laplace.cdf(-1.0),
        standard_laplace.cdf(0.0),
        standard_laplace.cdf(1.0),
        standard_laplace.cdf(3.0)
    );
    println!(
        "Shifted Laplace:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_laplace.cdf(-3.0),
        shifted_laplace.cdf(-1.0),
        shifted_laplace.cdf(0.0),
        shifted_laplace.cdf(1.0),
        shifted_laplace.cdf(3.0)
    );
    println!(
        "Narrow Laplace:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_laplace.cdf(-3.0),
        narrow_laplace.cdf(-1.0),
        narrow_laplace.cdf(0.0),
        narrow_laplace.cdf(1.0),
        narrow_laplace.cdf(3.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                      p=0.1     p=0.25    p=0.5     p=0.75    p=0.9");
    println!(
        "Standard Laplace:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_laplace.ppf(0.1)?,
        standard_laplace.ppf(0.25)?,
        standard_laplace.ppf(0.5)?,
        standard_laplace.ppf(0.75)?,
        standard_laplace.ppf(0.9)?
    );
    println!(
        "Shifted Laplace:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_laplace.ppf(0.1)?,
        shifted_laplace.ppf(0.25)?,
        shifted_laplace.ppf(0.5)?,
        shifted_laplace.ppf(0.75)?,
        shifted_laplace.ppf(0.9)?
    );
    println!(
        "Narrow Laplace:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_laplace.ppf(0.1)?,
        narrow_laplace.ppf(0.25)?,
        narrow_laplace.ppf(0.5)?,
        narrow_laplace.ppf(0.75)?,
        narrow_laplace.ppf(0.9)?
    );

    // Generate and display random samples
    println!("\nRandom Samples from Standard Laplace (using trait implementation):");
    let samples = Distribution::rvs(&standard_laplace, 10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Also demonstrate direct method
    println!("\nRandom Samples from Standard Laplace (using direct method):");
    let samples_direct = standard_laplace.rvs_vec(10)?;
    for (i, sample) in samples_direct.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Check the closeness of inverse CDF and CDF operations
    println!("\nVerifying the inverse relationship between CDF and PPF:");

    let test_values = [-2.0, -0.5, 0.0, 0.5, 2.0];
    for &x in &test_values {
        // Using direct methods
        let p = standard_laplace.cdf(x);
        let x_back = standard_laplace.ppf(p)?;
        let diff = f64::abs(x - x_back);
        println!(
            "  Direct: x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p, x_back, diff
        );

        // Using trait methods
        let p_trait = ContinuousDistribution::cdf(&standard_laplace, x);
        let x_back_trait = ContinuousDistribution::ppf(&standard_laplace, p_trait)?;
        let diff_trait = f64::abs(x - x_back_trait);
        println!(
            "  Trait:  x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p_trait, x_back_trait, diff_trait
        );
    }

    println!("\nLaplace Distribution Applications:");
    println!("1. Signal processing for robust signal detection");
    println!("2. Financial modeling for returns with higher kurtosis than normal");
    println!("3. Used in Bayesian inference as a prior (related to L1 regularization)");
    println!("4. Modeling error distributions in robust regression");
    println!("5. Image edge detection algorithms");
    println!("6. Speech recognition and natural language processing");

    println!("\nUnique Properties of the Laplace Distribution:");
    println!("1. Higher kurtosis (3) than normal distribution (0), more outliers");
    println!("2. 'Peakier' at the mode and heavier tails than normal");
    println!("3. Mean, median, and mode are equal (all equal to the location parameter)");
    println!("4. Variance is 2·scale²");
    println!("5. Related to the exponential distribution (absolute difference from loc)");
    println!("6. Maximum entropy distribution with fixed variance and zero mean");

    println!("\nKey formulas:");
    println!("- PDF: f(x) = (1/(2·scale))·exp(-|x-loc|/scale)");
    println!("- CDF: F(x) = 0.5·exp((x-loc)/scale) if x < loc");
    println!("       F(x) = 1 - 0.5·exp(-(x-loc)/scale) if x ≥ loc");
    println!("- Quantile: Q(p) = loc + scale·ln(2p) if p < 0.5");
    println!("            Q(p) = loc - scale·ln(2(1-p)) if p ≥ 0.5");
    println!("- Mean/Median/Mode = loc");
    println!("- Variance = 2·scale²");
    println!("- Entropy = 1 + ln(2·scale)");

    // Demonstrate trait-based interfaces vs direct methods
    println!("\nTrait-based interfaces:");
    println!("Using Distribution trait:");
    println!("  Mean (trait): {}", Distribution::mean(&standard_laplace));
    println!("  Mean (direct): {}", standard_laplace.mean());
    println!(
        "  Variance (trait): {}",
        Distribution::var(&standard_laplace)
    );
    println!("  Variance (direct): {}", standard_laplace.var());
    println!(
        "  Standard deviation (trait): {}",
        Distribution::std(&standard_laplace)
    );
    println!("  Standard deviation (direct): {}", standard_laplace.std());
    println!(
        "  Entropy (trait): {}",
        Distribution::entropy(&standard_laplace)
    );
    println!("  Entropy (direct): {}", standard_laplace.entropy());

    println!("\nUsing ContinuousDistribution trait:");
    println!(
        "  PDF at x=0 (trait): {}",
        ContinuousDistribution::pdf(&standard_laplace, 0.0)
    );
    println!("  PDF at x=0 (direct): {}", standard_laplace.pdf(0.0));
    println!(
        "  CDF at x=0 (trait): {}",
        ContinuousDistribution::cdf(&standard_laplace, 0.0)
    );
    println!("  CDF at x=0 (direct): {}", standard_laplace.cdf(0.0));

    let p = 0.75;
    let q_trait = ContinuousDistribution::ppf(&standard_laplace, p)?;
    let q_direct = standard_laplace.ppf(p)?;
    println!("  PPF at p=0.75 (trait): {}", q_trait);
    println!("  PPF at p=0.75 (direct): {}", q_direct);

    let sf_trait = 1.0 - ContinuousDistribution::cdf(&standard_laplace, 0.0);
    println!("  Survival function at x=0 (trait): {}", sf_trait);
    println!(
        "  Survival function computed manually: {}",
        1.0 - standard_laplace.cdf(0.0)
    );

    println!("\nAdvantage of trait system: Consistent interface for working with different distributions");

    Ok(())
}
