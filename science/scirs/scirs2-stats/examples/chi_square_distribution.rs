use scirs2_stats::distributions::ChiSquare;
use scirs2_stats::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use statrs::statistics::Statistics;
use std::f64;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chi-Square Distribution Example");
    println!("============================\n");

    // Create a chi-square distribution with 2 degrees of freedom
    let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0)?;

    println!("1. Chi-Square Distribution (df=2)");
    println!("   Mean: {}", chi2.mean());
    println!("   Variance: {}", chi2.var());
    println!("   Standard deviation: {}\n", chi2.std());

    // Using direct methods
    println!("2. PDF and CDF at various points (direct method):");
    println!("   PDF(1.0) = {:.6}", chi2.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", chi2.pdf(2.0));
    println!("   PDF(3.0) = {:.6}", chi2.pdf(3.0));
    println!("   CDF(1.0) = {:.6}", chi2.cdf(1.0));
    println!("   CDF(2.0) = {:.6}", chi2.cdf(2.0));
    println!("   CDF(3.0) = {:.6}\n", chi2.cdf(3.0));

    // Using the trait interface (ContinuousDistribution)
    let dist: &dyn ContinuousDistribution<f64> = &chi2;

    println!("3. PDF and CDF at various points (trait method):");
    println!("   PDF(1.0) = {:.6}", dist.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", dist.pdf(2.0));
    println!("   PDF(3.0) = {:.6}", dist.pdf(3.0));
    println!("   CDF(1.0) = {:.6}", dist.cdf(1.0));
    println!("   CDF(2.0) = {:.6}", dist.cdf(2.0));
    println!("   CDF(3.0) = {:.6}\n", dist.cdf(3.0));

    // Note: ChiSquare doesn't implement ContinuousCDF trait
    // so survival function, hazard function etc. are not available
    println!("4. Additional functions:");
    println!(
        "   Manual survival function SF(2.0) = {:.6}",
        1.0 - dist.cdf(2.0)
    );

    // Quantile function (inverse CDF)
    println!("5. Quantile functions and critical values:");
    println!("   Median (p=0.50) = {:.6}", dist.ppf(0.50)?);
    println!(
        "   Critical value for α=0.05 (p=0.95) = {:.6}",
        dist.ppf(0.95)?
    );
    println!(
        "   Critical value for α=0.01 (p=0.99) = {:.6}",
        dist.ppf(0.99)?
    );
    // Note: ISF not available without ContinuousCDF trait implementation

    // Random sampling
    println!("6. Random sampling:");
    let samples = dist.rvs(5)?;
    println!("   5 random samples: {:?}\n", samples);

    // Other degrees of freedom
    println!("7. Comparing different degrees of freedom:");

    // Chi-square with 1 degree of freedom
    let chi1 = ChiSquare::new(1.0f64, 0.0, 1.0)?;
    println!("   χ²(1) distribution:");
    println!("   Mean: {}", chi1.mean());
    println!("   Variance: {}", chi1.var());
    println!("   Critical value (α=0.05): {:.6}", chi1.ppf(0.95)?);
    println!("   PDF(1.0) = {:.6}\n", chi1.pdf(1.0));

    // Chi-square with 5 degrees of freedom
    let chi5 = ChiSquare::new(5.0f64, 0.0, 1.0)?;
    println!("   χ²(5) distribution:");
    println!("   Mean: {}", chi5.mean());
    println!("   Variance: {}", chi5.var());
    println!("   Critical value (α=0.05): {:.6}", chi5.ppf(0.95)?);
    println!("   PDF(5.0) = {:.6}\n", chi5.pdf(5.0));

    // Non-standard chi-square distribution
    let custom = ChiSquare::new(3.0f64, 1.0, 2.0)?;
    println!("8. Non-standard chi-square (df=3, loc=1.0, scale=2.0):");
    println!("   Mean: {}", custom.mean());
    println!("   Variance: {}", custom.var());
    println!("   PDF(7.0) = {:.6}", custom.pdf(7.0));
    println!("   CDF(7.0) = {:.6}", custom.cdf(7.0));

    Ok(())
}
