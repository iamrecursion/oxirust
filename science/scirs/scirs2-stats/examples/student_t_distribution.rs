use scirs2_stats::distributions::student_t::StudentT;
use scirs2_stats::traits::{ContinuousCDF, ContinuousDistribution, Distribution};
use statrs::statistics::Statistics;
use std::f64;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Student's t Distribution Example");
    println!("==============================\n");

    // Create a t-distribution with 5 degrees of freedom
    let t5 = StudentT::new(5.0f64, 0.0, 1.0)?;

    println!("1. Student's t Distribution (df=5)");
    println!("   Mean: {}", t5.mean());
    println!("   Variance: {}", t5.var());
    println!("   Standard deviation: {}\n", t5.std());

    // Using direct methods
    println!("2. PDF and CDF at various points (direct method):");
    println!("   PDF(0.0) = {:.6}", t5.pdf(0.0));
    println!("   PDF(1.0) = {:.6}", t5.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", t5.pdf(2.0));
    println!("   CDF(0.0) = {:.6}", t5.cdf(0.0));
    println!("   CDF(1.0) = {:.6}", t5.cdf(1.0));
    println!("   CDF(2.0) = {:.6}\n", t5.cdf(2.0));

    // Using the trait interface (ContinuousDistribution)
    let dist: &dyn ContinuousCDF<f64> = &t5;

    println!("3. PDF and CDF at various points (trait method):");
    println!("   PDF(0.0) = {:.6}", dist.pdf(0.0));
    println!("   PDF(1.0) = {:.6}", dist.pdf(1.0));
    println!("   PDF(2.0) = {:.6}", dist.pdf(2.0));
    println!("   CDF(0.0) = {:.6}", dist.cdf(0.0));
    println!("   CDF(1.0) = {:.6}", dist.cdf(1.0));
    println!("   CDF(2.0) = {:.6}\n", dist.cdf(2.0));

    // Survival function and hazard function
    println!("4. Additional functions available through the trait:");
    println!("   Survival function SF(1.0) = {:.6}", dist.sf(1.0));
    println!("   Hazard function h(1.0) = {:.6}", dist.hazard(1.0));
    println!(
        "   Cumulative hazard function H(1.0) = {:.6}\n",
        dist.cumhazard(1.0)
    );

    // Quantile function (inverse CDF)
    println!("5. Quantile functions:");
    println!("   Median (p=0.50) = {:.6}", dist.ppf(0.50)?);
    println!("   95th percentile (p=0.95) = {:.6}", dist.ppf(0.95)?);
    println!("   99th percentile (p=0.99) = {:.6}", dist.ppf(0.99)?);
    println!(
        "   Inverse survival function (p=0.95) = {:.6}\n",
        dist.isf(0.95)?
    );

    // Random sampling
    println!("6. Random sampling:");
    let samples = dist.rvs(5)?;
    println!("   5 random samples: {:?}\n", samples);

    // Other degrees of freedom
    println!("7. Comparing different degrees of freedom:");

    // t-distribution with 1 degree of freedom (Cauchy distribution)
    let t1 = StudentT::new(1.0f64, 0.0, 1.0)?;
    println!("   t(1) [Cauchy distribution]:");
    println!("   Mean: {} (undefined)", t1.mean());
    println!("   Variance: {} (undefined)", t1.var());
    println!("   PDF(0.0) = {:.6}", t1.pdf(0.0));
    println!("   PDF(1.0) = {:.6}\n", t1.pdf(1.0));

    // t-distribution with 30 degrees of freedom (close to normal)
    let t30 = StudentT::new(30.0f64, 0.0, 1.0)?;
    println!("   t(30) [Close to normal distribution]:");
    println!("   Mean: {}", t30.mean());
    println!("   Variance: {:.6}", t30.var());
    println!("   PDF(0.0) = {:.6}", t30.pdf(0.0));
    println!("   PDF(1.0) = {:.6}\n", t30.pdf(1.0));

    // Non-standard t-distribution
    let custom = StudentT::new(5.0f64, 1.0, 2.0)?;
    println!("8. Non-standard t-distribution (df=5, loc=1.0, scale=2.0):");
    println!("   Mean: {}", custom.mean());
    println!("   PDF(1.0) = {:.6}", custom.pdf(1.0));
    println!("   CDF(1.0) = {:.6}", custom.cdf(1.0));

    Ok(())
}
