use scirs2_stats::distributions::cauchy::Cauchy;
use scirs2_stats::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Cauchy (Lorentz) Distribution Example");
    println!("------------------------------------");

    // Create Cauchy distributions with different parameters
    let standard_cauchy = Cauchy::new(0.0, 1.0)?; // Standard Cauchy (loc=0, scale=1)
    let shifted_cauchy = Cauchy::new(2.0, 1.0)?; // Shifted to location=2
    let narrow_cauchy = Cauchy::new(0.0, 0.5)?; // Narrower with scale=0.5
    let _custom_cauchy = Cauchy::new(-1.0, 2.0)?; // Custom with loc=-1, scale=2

    // Print parameters and properties
    println!("\n1. Standard Cauchy (loc=0, scale=1)");
    println!("   Location parameter: {}", standard_cauchy.loc);
    println!("   Scale parameter: {}", standard_cauchy.scale);
    println!("   Median: {}", standard_cauchy.median());
    println!("   Mode: {}", standard_cauchy.mode());
    println!("   Interquartile range (IQR): {}", standard_cauchy.iqr());
    println!("   Entropy: {:.7}", standard_cauchy.entropy());
    println!(
        "   Mean: {} (undefined, returns NaN)",
        Distribution::mean(&standard_cauchy)
    );
    println!(
        "   Variance: {} (undefined, returns NaN)",
        Distribution::var(&standard_cauchy)
    );

    println!("\n2. Shifted Cauchy (loc=2, scale=1)");
    println!("   Location parameter: {}", shifted_cauchy.loc);
    println!("   Scale parameter: {}", shifted_cauchy.scale);
    println!("   Median: {}", shifted_cauchy.median());
    println!("   Mode: {}", shifted_cauchy.mode());
    println!("   Interquartile range (IQR): {}", shifted_cauchy.iqr());
    println!("   Entropy: {:.7}", shifted_cauchy.entropy());

    println!("\n3. Narrow Cauchy (loc=0, scale=0.5)");
    println!("   Location parameter: {}", narrow_cauchy.loc);
    println!("   Scale parameter: {}", narrow_cauchy.scale);
    println!("   Median: {}", narrow_cauchy.median());
    println!("   Mode: {}", narrow_cauchy.mode());
    println!("   Interquartile range (IQR): {}", narrow_cauchy.iqr());
    println!("   Entropy: {:.7}", narrow_cauchy.entropy());

    // Calculate PDF values at different points
    println!("\nPDF Values:");
    println!("                     x=-3      x=-1      x=0       x=1       x=3");
    println!(
        "Standard Cauchy:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_cauchy.pdf(-3.0),
        standard_cauchy.pdf(-1.0),
        standard_cauchy.pdf(0.0),
        standard_cauchy.pdf(1.0),
        standard_cauchy.pdf(3.0)
    );
    println!(
        "Shifted Cauchy:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_cauchy.pdf(-3.0),
        shifted_cauchy.pdf(-1.0),
        shifted_cauchy.pdf(0.0),
        shifted_cauchy.pdf(1.0),
        shifted_cauchy.pdf(3.0)
    );
    println!(
        "Narrow Cauchy:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_cauchy.pdf(-3.0),
        narrow_cauchy.pdf(-1.0),
        narrow_cauchy.pdf(0.0),
        narrow_cauchy.pdf(1.0),
        narrow_cauchy.pdf(3.0)
    );

    // Calculate CDF values at different points
    println!("\nCDF Values:");
    println!("                     x=-3      x=-1      x=0       x=1       x=3");
    println!(
        "Standard Cauchy:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_cauchy.cdf(-3.0),
        standard_cauchy.cdf(-1.0),
        standard_cauchy.cdf(0.0),
        standard_cauchy.cdf(1.0),
        standard_cauchy.cdf(3.0)
    );
    println!(
        "Shifted Cauchy:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_cauchy.cdf(-3.0),
        shifted_cauchy.cdf(-1.0),
        shifted_cauchy.cdf(0.0),
        shifted_cauchy.cdf(1.0),
        shifted_cauchy.cdf(3.0)
    );
    println!(
        "Narrow Cauchy:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_cauchy.cdf(-3.0),
        narrow_cauchy.cdf(-1.0),
        narrow_cauchy.cdf(0.0),
        narrow_cauchy.cdf(1.0),
        narrow_cauchy.cdf(3.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                     p=0.1     p=0.25    p=0.5     p=0.75    p=0.9");
    println!(
        "Standard Cauchy:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        standard_cauchy.ppf(0.1)?,
        standard_cauchy.ppf(0.25)?,
        standard_cauchy.ppf(0.5)?,
        standard_cauchy.ppf(0.75)?,
        standard_cauchy.ppf(0.9)?
    );
    println!(
        "Shifted Cauchy:     {:.7} {:.7} {:.7} {:.7} {:.7}",
        shifted_cauchy.ppf(0.1)?,
        shifted_cauchy.ppf(0.25)?,
        shifted_cauchy.ppf(0.5)?,
        shifted_cauchy.ppf(0.75)?,
        shifted_cauchy.ppf(0.9)?
    );
    println!(
        "Narrow Cauchy:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        narrow_cauchy.ppf(0.1)?,
        narrow_cauchy.ppf(0.25)?,
        narrow_cauchy.ppf(0.5)?,
        narrow_cauchy.ppf(0.75)?,
        narrow_cauchy.ppf(0.9)?
    );

    // Generate and display random samples
    println!("\nRandom Samples from Standard Cauchy (using trait implementation):");
    let samples = Distribution::rvs(&standard_cauchy, 10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Also demonstrate direct method
    println!("\nRandom Samples from Standard Cauchy (using direct method):");
    let samples_direct = standard_cauchy.rvs_vec(10)?;
    for (i, sample) in samples_direct.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Check the closeness of inverse CDF and CDF operations
    println!("\nVerifying the inverse relationship between CDF and PPF:");

    let test_values = [-2.0, -0.5, 0.0, 0.5, 2.0];
    for &x in &test_values {
        // Using direct methods
        let p = standard_cauchy.cdf(x);
        let x_back = standard_cauchy.ppf(p)?;
        let diff = f64::abs(x - x_back);
        println!(
            "  Direct: x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p, x_back, diff
        );

        // Using trait methods
        let p_trait = ContinuousDistribution::cdf(&standard_cauchy, x);
        let x_back_trait = ContinuousDistribution::ppf(&standard_cauchy, p_trait)?;
        let diff_trait = f64::abs(x - x_back_trait);
        println!(
            "  Trait:  x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p_trait, x_back_trait, diff_trait
        );
    }

    println!("\nCauchy Distribution Applications:");
    println!("1. Modeling resonance behavior in physics (e.g., forced oscillators)");
    println!("2. Describing spectral line shapes in spectroscopy");
    println!("3. Used in electrical engineering for impedance calculations");
    println!("4. Modeling heavy-tailed phenomena where outliers are common");
    println!("5. Used in signal processing for robust estimation");

    println!("\nUnique Properties of the Cauchy Distribution:");
    println!("1. Mean and higher moments are undefined (do not exist)");
    println!("2. Stable distribution: sum of Cauchy random variables is also Cauchy");
    println!("3. Heavy tails: probability of extreme values is much higher than normal");
    println!("4. Self-similar: ratio of two standard normal variables follows Cauchy");
    println!("5. Location parameter equals median, mode, and maximum likelihood estimate");

    println!("\nKey formulas:");
    println!("- PDF: f(x) = 1/(π·scale·(1 + ((x-loc)/scale)²))");
    println!("- CDF: F(x) = 0.5 + (1/π)·arctan((x-loc)/scale)");
    println!("- Quantile: Q(p) = loc + scale·tan(π·(p-0.5))");
    println!("- Median/Mode = loc");
    println!("- IQR = 2·scale");
    println!("- Entropy = log(4π·scale)");

    // Demonstrate trait-based interfaces
    println!("\nTrait-based interfaces:");
    println!("Using Distribution trait:");
    println!(
        "  Mean (trait): {} (undefined)",
        Distribution::mean(&standard_cauchy)
    );
    println!(
        "  Variance (trait): {} (undefined)",
        Distribution::var(&standard_cauchy)
    );
    println!(
        "  Standard deviation (trait): {} (undefined)",
        Distribution::std(&standard_cauchy)
    );
    println!(
        "  Entropy (trait): {}",
        Distribution::entropy(&standard_cauchy)
    );
    println!("  Entropy (direct): {}", standard_cauchy.entropy());

    println!("\nUsing ContinuousDistribution trait:");
    println!(
        "  PDF at x=0 (trait): {}",
        ContinuousDistribution::pdf(&standard_cauchy, 0.0)
    );
    println!("  PDF at x=0 (direct): {}", standard_cauchy.pdf(0.0));
    println!(
        "  CDF at x=0 (trait): {}",
        ContinuousDistribution::cdf(&standard_cauchy, 0.0)
    );
    println!("  CDF at x=0 (direct): {}", standard_cauchy.cdf(0.0));

    let p = 0.75;
    let q_trait = ContinuousDistribution::ppf(&standard_cauchy, p)?;
    let q_direct = standard_cauchy.ppf(p)?;
    println!("  PPF at p=0.75 (trait): {}", q_trait);
    println!("  PPF at p=0.75 (direct): {}", q_direct);

    // Note: Cauchy distribution doesn't implement ContinuousCDF trait
    // so we calculate survival function manually
    let sf_manual = 1.0 - standard_cauchy.cdf(0.0);
    println!("  Survival function at x=0 (manual): {}", sf_manual);

    println!("\nAdvantage of trait system:");
    println!("- Consistent interface for working with different distributions");
    println!("- Handle special cases appropriately (like undefined moments in Cauchy)");
    println!("- Derived methods (like survival function) come for free");

    Ok(())
}
