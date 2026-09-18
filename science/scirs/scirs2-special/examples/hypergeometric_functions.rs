use scirs2_special::error::SpecialResult;
use scirs2_special::{hyp1f1, hyp2f1, pochhammer};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> SpecialResult<()> {
    println!("Hypergeometric Functions Example");
    println!("===============================\n");

    // Pochhammer symbol examples
    println!("Pochhammer symbol (Rising factorial):");
    println!("------------------------------------");
    println!("(1)₄ = {}", pochhammer(1.0, 4));
    println!("(3)₂ = {}", pochhammer(3.0, 2));
    println!("(2)₃ = {}", pochhammer(2.0, 3));
    println!("(0.5)₂ = {}", pochhammer(0.5, 2));
    println!();

    // 1F1 (Confluent hypergeometric function) examples
    println!("1F1 (Confluent hypergeometric function):");
    println!("-------------------------------------");

    // Known values
    let a = 1.0;
    let b = 2.0;
    let z = 0.5;
    let start = Instant::now();
    let result = hyp1f1(a, b, z)?;
    let duration = start.elapsed();
    println!(
        "1F1({}, {}; {}) = {} (calculated in {:?})",
        a, b, z, result, duration
    );

    // Relation to error function
    let z = 1.0;
    let erf_z = 2.0 / PI.sqrt() * hyp1f1(0.5, 1.5, -z * z)? * z;
    println!("erf({}) ≈ {} (via 1F1)", z, erf_z);

    // Relation to Bessel functions
    let z = 2.0;
    let bessel_i0 = hyp1f1(0.5, 1.0, z * z / 4.0)?;
    println!("I₀({}) ≈ {} (via 1F1)", z, bessel_i0);

    println!();

    // 2F1 (Gauss hypergeometric function) examples
    println!("2F1 (Gauss hypergeometric function):");
    println!("----------------------------------");

    // Some standard values
    let a = 1.0;
    let b = 2.0;
    let c = 3.0;
    let z = 0.5;
    let start = Instant::now();
    let result = hyp2f1(a, b, c, z)?;
    let duration = start.elapsed();
    println!(
        "2F1({}, {}, {}; {}) = {} (calculated in {:?})",
        a, b, c, z, result, duration
    );

    // Special cases
    println!(
        "2F1(1, 1, 2; 0.5) = {} (= 2·ln(2))",
        hyp2f1(1.0, 1.0, 2.0, 0.5)?
    );
    println!(
        "2F1(-3, 2, 1; 0.5) = {} (polynomial case)",
        hyp2f1(-3.0, 2.0, 1.0, 0.5)?
    );

    // Relation to complete elliptic integral of the first kind
    let m = 0.5; // parameter between 0 and 1
    let k = hyp2f1(0.5, 0.5, 1.0, m)? * PI / 2.0;
    println!("Complete elliptic integral K({}) ≈ {} (via 2F1)", m, k);

    println!();

    // Demonstrate mathematical identities
    println!("Mathematical Identities:");
    println!("----------------------");

    // Kummer's transformation for 1F1
    let a = 0.5f64;
    let b = 1.5;
    let z = 0.75;
    let lhs = hyp1f1(a, b, z)?;
    let rhs = z.exp() * hyp1f1(b - a, b, -z)?;
    println!(
        "Kummer's transformation: 1F1({}, {}; {}) = {}",
        a, b, z, lhs
    );
    println!(
        "                          e^z·1F1({}-{}, {}; -{}) = {}",
        b, a, b, z, rhs
    );
    println!("Relative difference: {:.2e}", (lhs - rhs).abs() / lhs.abs());

    // Symmetry in parameters for 2F1
    let a = 0.5f64;
    let b = 1.5;
    let c = 2.5;
    let z = 0.25;
    let val1 = hyp2f1(a, b, c, z)?;
    let val2 = hyp2f1(b, a, c, z)?;
    println!(
        "Symmetry in parameters: 2F1({}, {}, {}; {}) = {}",
        a, b, c, z, val1
    );
    println!(
        "                         2F1({}, {}, {}; {}) = {}",
        b, a, c, z, val2
    );
    println!(
        "Relative difference: {:.2e}",
        (val1 - val2).abs() / val1.abs()
    );

    Ok(())
}
