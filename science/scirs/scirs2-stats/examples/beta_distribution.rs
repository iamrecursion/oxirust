use scirs2_stats::distributions::beta::Beta;
use scirs2_stats::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Beta Distribution Example");
    println!("-----------------------");

    // Create Beta distributions with different parameters
    let uniform_beta = Beta::new(1.0, 1.0, 0.0, 1.0)?; // Uniform distribution
    let symmetric_beta = Beta::new(2.0, 2.0, 0.0, 1.0)?; // Symmetric bell-shaped
    let skewed_beta = Beta::new(2.0, 5.0, 0.0, 1.0)?; // Right-skewed
    let left_skewed_beta = Beta::new(5.0, 2.0, 0.0, 1.0)?; // Left-skewed
    let shifted_beta = Beta::new(2.0, 2.0, 1.0, 2.0)?; // Shifted and scaled

    // Print parameters and properties
    println!("\n1. Uniform Beta (alpha=1, beta=1)");
    println!("   Alpha: {}", uniform_beta.alpha);
    println!("   Beta: {}", uniform_beta.beta);
    println!("   Location: {}", uniform_beta.loc);
    println!("   Scale: {}", uniform_beta.scale);
    println!("   Mean: {}", Distribution::mean(&uniform_beta));
    println!("   Variance: {}", Distribution::var(&uniform_beta));
    println!(
        "   Standard deviation: {:.6}",
        Distribution::std(&uniform_beta)
    );
    println!("   Entropy: {:.6}", Distribution::entropy(&uniform_beta));

    println!("\n2. Symmetric Beta (alpha=2, beta=2)");
    println!("   Alpha: {}", symmetric_beta.alpha);
    println!("   Beta: {}", symmetric_beta.beta);
    println!("   Mean: {}", Distribution::mean(&symmetric_beta));
    println!("   Variance: {}", Distribution::var(&symmetric_beta));
    println!(
        "   Standard deviation: {:.6}",
        Distribution::std(&symmetric_beta)
    );

    println!("\n3. Skewed Beta (alpha=2, beta=5)");
    println!("   Alpha: {}", skewed_beta.alpha);
    println!("   Beta: {}", skewed_beta.beta);
    println!("   Mean: {}", Distribution::mean(&skewed_beta));
    println!("   Variance: {}", Distribution::var(&skewed_beta));
    println!(
        "   Standard deviation: {:.6}",
        Distribution::std(&skewed_beta)
    );

    println!("\n4. Left-Skewed Beta (alpha=5, beta=2)");
    println!("   Alpha: {}", left_skewed_beta.alpha);
    println!("   Beta: {}", left_skewed_beta.beta);
    println!("   Mean: {}", Distribution::mean(&left_skewed_beta));
    println!("   Variance: {}", Distribution::var(&left_skewed_beta));
    println!(
        "   Standard deviation: {:.6}",
        Distribution::std(&left_skewed_beta)
    );

    println!("\n5. Shifted Beta (alpha=2, beta=2, loc=1, scale=2)");
    println!("   Alpha: {}", shifted_beta.alpha);
    println!("   Beta: {}", shifted_beta.beta);
    println!("   Location: {}", shifted_beta.loc);
    println!("   Scale: {}", shifted_beta.scale);
    println!("   Mean: {}", Distribution::mean(&shifted_beta));
    println!("   Variance: {}", Distribution::var(&shifted_beta));
    println!(
        "   Standard deviation: {:.6}",
        Distribution::std(&shifted_beta)
    );

    // Calculate PDF values at different points
    println!("\nPDF Values:");
    println!("                     x=0       x=0.2     x=0.5     x=0.8     x=1.0");
    println!(
        "Uniform Beta:        {:.7} {:.7} {:.7} {:.7} {:.7}",
        uniform_beta.pdf(0.0),
        uniform_beta.pdf(0.2),
        uniform_beta.pdf(0.5),
        uniform_beta.pdf(0.8),
        uniform_beta.pdf(1.0)
    );
    println!(
        "Symmetric Beta:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        symmetric_beta.pdf(0.0),
        symmetric_beta.pdf(0.2),
        symmetric_beta.pdf(0.5),
        symmetric_beta.pdf(0.8),
        symmetric_beta.pdf(1.0)
    );
    println!(
        "Skewed Beta:         {:.7} {:.7} {:.7} {:.7} {:.7}",
        skewed_beta.pdf(0.0),
        skewed_beta.pdf(0.2),
        skewed_beta.pdf(0.5),
        skewed_beta.pdf(0.8),
        skewed_beta.pdf(1.0)
    );
    println!(
        "Left-Skewed Beta:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        left_skewed_beta.pdf(0.0),
        left_skewed_beta.pdf(0.2),
        left_skewed_beta.pdf(0.5),
        left_skewed_beta.pdf(0.8),
        left_skewed_beta.pdf(1.0)
    );

    // Calculate CDF values at different points
    println!("\nCDF Values:");
    println!("                     x=0       x=0.2     x=0.5     x=0.8     x=1.0");
    println!(
        "Uniform Beta:        {:.7} {:.7} {:.7} {:.7} {:.7}",
        uniform_beta.cdf(0.0),
        uniform_beta.cdf(0.2),
        uniform_beta.cdf(0.5),
        uniform_beta.cdf(0.8),
        uniform_beta.cdf(1.0)
    );
    println!(
        "Symmetric Beta:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        symmetric_beta.cdf(0.0),
        symmetric_beta.cdf(0.2),
        symmetric_beta.cdf(0.5),
        symmetric_beta.cdf(0.8),
        symmetric_beta.cdf(1.0)
    );
    println!(
        "Skewed Beta:         {:.7} {:.7} {:.7} {:.7} {:.7}",
        skewed_beta.cdf(0.0),
        skewed_beta.cdf(0.2),
        skewed_beta.cdf(0.5),
        skewed_beta.cdf(0.8),
        skewed_beta.cdf(1.0)
    );
    println!(
        "Left-Skewed Beta:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        left_skewed_beta.cdf(0.0),
        left_skewed_beta.cdf(0.2),
        left_skewed_beta.cdf(0.5),
        left_skewed_beta.cdf(0.8),
        left_skewed_beta.cdf(1.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                     p=0.1     p=0.25    p=0.5     p=0.75    p=0.9");
    println!(
        "Uniform Beta:        {:.7} {:.7} {:.7} {:.7} {:.7}",
        uniform_beta.ppf(0.1)?,
        uniform_beta.ppf(0.25)?,
        uniform_beta.ppf(0.5)?,
        uniform_beta.ppf(0.75)?,
        uniform_beta.ppf(0.9)?
    );
    println!(
        "Symmetric Beta:      {:.7} {:.7} {:.7} {:.7} {:.7}",
        symmetric_beta.ppf(0.1)?,
        symmetric_beta.ppf(0.25)?,
        symmetric_beta.ppf(0.5)?,
        symmetric_beta.ppf(0.75)?,
        symmetric_beta.ppf(0.9)?
    );
    println!(
        "Skewed Beta:         {:.7} {:.7} {:.7} {:.7} {:.7}",
        skewed_beta.ppf(0.1)?,
        skewed_beta.ppf(0.25)?,
        skewed_beta.ppf(0.5)?,
        skewed_beta.ppf(0.75)?,
        skewed_beta.ppf(0.9)?
    );
    println!(
        "Left-Skewed Beta:    {:.7} {:.7} {:.7} {:.7} {:.7}",
        left_skewed_beta.ppf(0.1)?,
        left_skewed_beta.ppf(0.25)?,
        left_skewed_beta.ppf(0.5)?,
        left_skewed_beta.ppf(0.75)?,
        left_skewed_beta.ppf(0.9)?
    );

    // Generate random samples
    println!("\nRandom Samples from Symmetric Beta (using trait implementation):");
    let samples = Distribution::rvs(&symmetric_beta, 10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Generate random samples using direct method
    println!("\nRandom Samples from Symmetric Beta (using direct method):");
    let samples_direct = symmetric_beta.rvs_vec(10)?;
    for (i, sample) in samples_direct.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Check inverse relationship
    println!("\nVerifying the inverse relationship between CDF and PPF:");
    let test_values = [0.2, 0.5, 0.8];
    for &x in &test_values {
        // Using direct methods
        let p = symmetric_beta.cdf(x);
        let x_back = symmetric_beta.ppf(p)?;
        let diff = f64::abs(x - x_back);
        println!(
            "  Direct: x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p, x_back, diff
        );

        // Using trait methods
        let p_trait = ContinuousDistribution::cdf(&symmetric_beta, x);
        let x_back_trait = ContinuousDistribution::ppf(&symmetric_beta, p_trait)?;
        let diff_trait = f64::abs(x - x_back_trait);
        println!(
            "  Trait:  x = {:.4}, CDF(x) = {:.4}, PPF(CDF(x)) = {:.4}, Difference = {:.1e}",
            x, p_trait, x_back_trait, diff_trait
        );
    }

    println!("\nBeta Distribution Applications:");
    println!("1. Bayesian statistics (as a conjugate prior)");
    println!("2. Modeling proportions and percentages");
    println!("3. Modeling random variables that are constrained to a finite interval");
    println!("4. Project management and PERT analysis");
    println!("5. Expert opinion elicitation");
    println!("6. Reliability analysis");

    println!("\nTrait-based interfaces:");
    println!("Using Distribution trait:");
    println!("  Mean (trait): {}", Distribution::mean(&symmetric_beta));
    println!("  Mean (direct): N/A (not implemented in base class)");
    println!("  Variance (trait): {}", Distribution::var(&symmetric_beta));
    println!(
        "  Standard deviation (trait): {}",
        Distribution::std(&symmetric_beta)
    );
    println!(
        "  Entropy (trait): {}",
        Distribution::entropy(&symmetric_beta)
    );

    println!("\nUsing ContinuousDistribution trait:");
    println!(
        "  PDF at x=0.5 (trait): {}",
        ContinuousDistribution::pdf(&symmetric_beta, 0.5)
    );
    println!("  PDF at x=0.5 (direct): {}", symmetric_beta.pdf(0.5));
    println!(
        "  CDF at x=0.5 (trait): {}",
        ContinuousDistribution::cdf(&symmetric_beta, 0.5)
    );
    println!("  CDF at x=0.5 (direct): {}", symmetric_beta.cdf(0.5));

    let p = 0.75;
    let q_trait = ContinuousDistribution::ppf(&symmetric_beta, p)?;
    let q_direct = symmetric_beta.ppf(p)?;
    println!("  PPF at p=0.75 (trait): {}", q_trait);
    println!("  PPF at p=0.75 (direct): {}", q_direct);

    let sf_trait = symmetric_beta.sf(0.5);
    println!("  Survival function at x=0.5 (trait): {}", sf_trait);
    println!(
        "  Survival function computed manually: {}",
        1.0 - symmetric_beta.cdf(0.5)
    );

    println!("\nAdvantage of trait system:");
    println!("- Consistent interface for working with different distributions");
    println!("- Automatic implementation of derived methods (sf, hazard, etc.)");
    println!("- Support for distribution-specific methods while maintaining compatibility");

    Ok(())
}
