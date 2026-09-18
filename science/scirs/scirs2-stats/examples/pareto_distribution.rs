use scirs2_stats::distributions::pareto::Pareto;
use statrs::statistics::Statistics;
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Pareto Distribution Example");
    println!("--------------------------");

    // Create Pareto distributions with different shape parameters
    let pareto1 = Pareto::new(1.0, 1.0, 0.0)?; // Shape = 1 (undefined mean)
    let pareto2 = Pareto::new(2.0, 1.0, 0.0)?; // Shape = 2 (undefined variance)
    let pareto3 = Pareto::new(3.0, 1.0, 0.0)?; // Shape = 3 (finite mean and variance)
    let pareto_custom = Pareto::new(2.5, 2.0, 1.0)?; // Custom with scale=2, loc=1

    // Print parameters for Pareto1
    println!("\n1. Standard Pareto (shape=1, scale=1, loc=0)");
    println!("   shape = {}", pareto1.shape);
    println!("   scale = {}", pareto1.scale);
    println!("   loc = {}", pareto1.loc);
    println!("   mean = {}", pareto1.mean()); // Will show infinity
    println!("   variance = {}", pareto1.var()); // Will show infinity
    println!("   median = {:.7}", pareto1.median());
    println!("   mode = {:.7}", pareto1.mode());

    // Print parameters for Pareto2
    println!("\n2. Pareto with shape=2 (scale=1, loc=0)");
    println!("   shape = {}", pareto2.shape);
    println!("   scale = {}", pareto2.scale);
    println!("   loc = {}", pareto2.loc);
    println!("   mean = {:.7}", pareto2.mean());
    println!("   variance = {}", pareto2.var()); // Will show infinity
    println!("   median = {:.7}", pareto2.median());
    println!("   mode = {:.7}", pareto2.mode());

    // Print parameters for Pareto3
    println!("\n3. Pareto with shape=3 (scale=1, loc=0)");
    println!("   shape = {}", pareto3.shape);
    println!("   scale = {}", pareto3.scale);
    println!("   loc = {}", pareto3.loc);
    println!("   mean = {:.7}", pareto3.mean());
    println!("   variance = {:.7}", pareto3.var());
    println!("   median = {:.7}", pareto3.median());
    println!("   mode = {:.7}", pareto3.mode());

    // Print parameters for custom Pareto
    println!("\n4. Custom Pareto (shape=2.5, scale=2, loc=1)");
    println!("   shape = {}", pareto_custom.shape);
    println!("   scale = {}", pareto_custom.scale);
    println!("   loc = {}", pareto_custom.loc);
    println!("   mean = {:.7}", pareto_custom.mean());
    println!("   variance = {:.7}", pareto_custom.var());
    println!("   median = {:.7}", pareto_custom.median());
    println!("   mode = {:.7}", pareto_custom.mode());

    // Calculate PDF values at different points
    println!("\nPDF Values:");
    println!("                     x=1.0     x=1.5     x=2.0     x=3.0");
    println!(
        "Standard Pareto:    {:.7} {:.7} {:.7} {:.7}",
        pareto1.pdf(1.0),
        pareto1.pdf(1.5),
        pareto1.pdf(2.0),
        pareto1.pdf(3.0)
    );
    println!(
        "Pareto (shape=2):   {:.7} {:.7} {:.7} {:.7}",
        pareto2.pdf(1.0),
        pareto2.pdf(1.5),
        pareto2.pdf(2.0),
        pareto2.pdf(3.0)
    );
    println!(
        "Pareto (shape=3):   {:.7} {:.7} {:.7} {:.7}",
        pareto3.pdf(1.0),
        pareto3.pdf(1.5),
        pareto3.pdf(2.0),
        pareto3.pdf(3.0)
    );

    // Calculate CDF values at different points
    println!("\nCDF Values:");
    println!("                     x=1.0     x=1.5     x=2.0     x=3.0");
    println!(
        "Standard Pareto:    {:.7} {:.7} {:.7} {:.7}",
        pareto1.cdf(1.0),
        pareto1.cdf(1.5),
        pareto1.cdf(2.0),
        pareto1.cdf(3.0)
    );
    println!(
        "Pareto (shape=2):   {:.7} {:.7} {:.7} {:.7}",
        pareto2.cdf(1.0),
        pareto2.cdf(1.5),
        pareto2.cdf(2.0),
        pareto2.cdf(3.0)
    );
    println!(
        "Pareto (shape=3):   {:.7} {:.7} {:.7} {:.7}",
        pareto3.cdf(1.0),
        pareto3.cdf(1.5),
        pareto3.cdf(2.0),
        pareto3.cdf(3.0)
    );

    // Calculate quantiles
    println!("\nQuantiles (Inverse CDF):");
    println!("                     p=0.1     p=0.5     p=0.9");
    println!(
        "Standard Pareto:    {:.7} {:.7} {:.7}",
        pareto1.ppf(0.1)?,
        pareto1.ppf(0.5)?,
        pareto1.ppf(0.9)?
    );
    println!(
        "Pareto (shape=2):   {:.7} {:.7} {:.7}",
        pareto2.ppf(0.1)?,
        pareto2.ppf(0.5)?,
        pareto2.ppf(0.9)?
    );
    println!(
        "Pareto (shape=3):   {:.7} {:.7} {:.7}",
        pareto3.ppf(0.1)?,
        pareto3.ppf(0.5)?,
        pareto3.ppf(0.9)?
    );

    // Generate and display random samples
    println!("\nRandom Samples (shape=3, scale=1, loc=0):");
    let samples = pareto3.rvs(10)?;
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: {:.7}", i + 1, sample);
    }

    // Calculate some statistics from the samples
    println!("\nSample Statistics (from 10 samples):");
    let sum: f64 = samples.iter().sum();
    let mean = sum / 10.0;
    println!("  Sample mean: {:.7}", mean);
    println!("  Theoretical mean: {:.7}", pareto3.mean());

    println!("\nPareto Distribution Applications:");
    println!("1. Income and wealth distribution (Pareto's 80-20 principle)");
    println!("2. File sizes in computer systems");
    println!("3. City population sizes");
    println!("4. Sizes of sand particles and meteorites");
    println!("5. Insurance claim sizes");

    println!("\nKey Properties:");
    println!("- Heavy-tailed distribution (tail decays like a power law)");
    println!("- For shape <= 1: Mean is undefined (infinite)");
    println!("- For shape <= 2: Variance is undefined (infinite)");
    println!("- Mode is always equal to the scale parameter");
    println!("- Characterized by the 80-20 rule (Pareto principle)");

    println!("\nKey formulas:");
    println!("- PDF: f(x) = (shape/scale) * (scale/x)^(shape+1)  for x >= scale");
    println!("- CDF: F(x) = 1 - (scale/x)^shape  for x >= scale");
    println!("- Mean: (shape*scale)/(shape-1)  for shape > 1");
    println!("- Variance: (scale^2 * shape)/((shape-1)^2 * (shape-2))  for shape > 2");
    println!("- Median: scale * 2^(1/shape)");
    println!("- Mode: scale");

    Ok(())
}
