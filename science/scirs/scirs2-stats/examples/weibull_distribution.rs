use scirs2_stats::distributions::weibull::Weibull;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Weibull Distribution Example");
    println!("---------------------------");

    // Create Weibull distributions with different shape parameters
    let weibull1 = Weibull::new(1.0, 1.0, 0.0)?; // Exponential distribution (shape=1)
    let weibull2 = Weibull::new(2.0, 1.0, 0.0)?; // Rayleigh distribution (shape=2)
    let weibull3 = Weibull::new(3.5, 1.0, 0.0)?; // More peaked distribution (shape=3.5)
    let weibull_custom = Weibull::new(2.0, 2.0, 1.0)?; // Custom with scale=2, loc=1

    // Print parameters
    println!("\n1. Exponential-like Weibull (shape=1, scale=1, loc=0)");
    println!("   shape = {}", weibull1.shape);
    println!("   scale = {}", weibull1.scale);
    println!("   loc = {}", weibull1.loc);
    println!("   mean = {:.7}", weibull1.mean());
    println!("   variance = {:.7}", weibull1.var());
    println!("   median = {:.7}", weibull1.median());
    println!("   mode = {:.7}", weibull1.mode());

    println!("\n2. Rayleigh-like Weibull (shape=2, scale=1, loc=0)");
    println!("   shape = {}", weibull2.shape);
    println!("   scale = {}", weibull2.scale);
    println!("   loc = {}", weibull2.loc);
    println!("   mean = {:.7}", weibull2.mean());
    println!("   variance = {:.7}", weibull2.var());
    println!("   median = {:.7}", weibull2.median());
    println!("   mode = {:.7}", weibull2.mode());

    println!("\n3. Peaked Weibull (shape=3.5, scale=1, loc=0)");
    println!("   shape = {}", weibull3.shape);
    println!("   scale = {}", weibull3.scale);
    println!("   loc = {}", weibull3.loc);
    println!("   mean = {:.7}", weibull3.mean());
    println!("   variance = {:.7}", weibull3.var());
    println!("   median = {:.7}", weibull3.median());
    println!("   mode = {:.7}", weibull3.mode());

    println!("\n4. Custom Weibull (shape=2, scale=2, loc=1)");
    println!("   shape = {}", weibull_custom.shape);
    println!("   scale = {}", weibull_custom.scale);
    println!("   loc = {}", weibull_custom.loc);
    println!("   mean = {:.7}", weibull_custom.mean());
    println!("   variance = {:.7}", weibull_custom.var());
    println!("   median = {:.7}", weibull_custom.median());
    println!("   mode = {:.7}", weibull_custom.mode());

    // Calculate PDF values at different points
    println!("\nPDF Values:");
    println!("                   x=0.1     x=0.5     x=1.0     x=2.0");
    println!(
        "Exponential-like: {:.7} {:.7} {:.7} {:.7}",
        weibull1.pdf(0.1),
        weibull1.pdf(0.5),
        weibull1.pdf(1.0),
        weibull1.pdf(2.0)
    );
    println!(
        "Rayleigh-like:    {:.7} {:.7} {:.7} {:.7}",
        weibull2.pdf(0.1),
        weibull2.pdf(0.5),
        weibull2.pdf(1.0),
        weibull2.pdf(2.0)
    );
    println!(
        "Peaked Weibull:   {:.7} {:.7} {:.7} {:.7}",
        weibull3.pdf(0.1),
        weibull3.pdf(0.5),
        weibull3.pdf(1.0),
        weibull3.pdf(2.0)
    );

    // Calculate CDF values at different points
    println!("\nCDF Values:");
    println!("                   x=0.1     x=0.5     x=1.0     x=2.0");
    println!(
        "Exponential-like: {:.7} {:.7} {:.7} {:.7}",
        weibull1.cdf(0.1),
        weibull1.cdf(0.5),
        weibull1.cdf(1.0),
        weibull1.cdf(2.0)
    );
    println!(
        "Rayleigh-like:    {:.7} {:.7} {:.7} {:.7}",
        weibull2.cdf(0.1),
        weibull2.cdf(0.5),
        weibull2.cdf(1.0),
        weibull2.cdf(2.0)
    );
    println!(
        "Peaked Weibull:   {:.7} {:.7} {:.7} {:.7}",
        weibull3.cdf(0.1),
        weibull3.cdf(0.5),
        weibull3.cdf(1.0),
        weibull3.cdf(2.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                   p=0.1     p=0.5     p=0.9");
    println!(
        "Exponential-like: {:.7} {:.7} {:.7}",
        weibull1.ppf(0.1)?,
        weibull1.ppf(0.5)?,
        weibull1.ppf(0.9)?
    );
    println!(
        "Rayleigh-like:    {:.7} {:.7} {:.7}",
        weibull2.ppf(0.1)?,
        weibull2.ppf(0.5)?,
        weibull2.ppf(0.9)?
    );
    println!(
        "Peaked Weibull:   {:.7} {:.7} {:.7}",
        weibull3.ppf(0.1)?,
        weibull3.ppf(0.5)?,
        weibull3.ppf(0.9)?
    );

    // Generate and show random samples
    println!("\nRandom Samples (shape=2, scale=1, loc=0):");
    let samples = weibull2.rvs(10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    println!("\nWeibull Distribution Applications:");
    println!("1. Reliability engineering for modeling failure rates");
    println!("2. Wind speed distributions in wind energy studies");
    println!("3. Material strength and fatigue life modeling");
    println!("4. Extreme value analysis in hydrology and meteorology");
    println!("5. Survival analysis in medicine and health sciences");

    println!("\nSpecial cases of the Weibull distribution:");
    println!("- When shape = 1: Equivalent to the Exponential distribution");
    println!("- When shape = 2: Equivalent to the Rayleigh distribution");
    println!("- When shape = 3.6: Approximates the Normal distribution");
    println!(
        "- As shape → ∞: Approaches a degenerate distribution centered at the scale parameter"
    );

    println!("\nKey formulas:");
    println!("- PDF: f(x) = (shape/scale) * (x/scale)^(shape-1) * exp(-(x/scale)^shape)");
    println!("- CDF: F(x) = 1 - exp(-(x/scale)^shape)");
    println!("- Mean: scale * Γ(1 + 1/shape)  [where Γ is the gamma function]");
    println!("- Variance: scale^2 * [Γ(1 + 2/shape) - (Γ(1 + 1/shape))^2]");
    println!("- Median: scale * (ln(2))^(1/shape)");
    println!("- Mode: scale * ((shape-1)/shape)^(1/shape)  [for shape > 1]");

    Ok(())
}
