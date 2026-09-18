// Simple test to verify implementations
use scirs2_core::Complex64;
use scirs2_special::*;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing implemented functions...");

    // Test Wright Bessel function
    let result = wright_bessel(1.0, 1.0, 0.0)?;
    println!("Wright Bessel(1.0, 1.0, 0.0) = {}", result);
    assert!((result - 1.0).abs() < 1e-10, "Wright Bessel test failed");

    // Test Wright Bessel complex function
    let result_complex =
        wright_bessel_complex(1.0, Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))?;
    println!(
        "Wright Bessel complex(1.0, 1+0i, 0+0i) = {}",
        result_complex
    );
    assert!(
        (result_complex.re - 1.0).abs() < 1e-10,
        "Wright Bessel complex test failed"
    );
    assert!(
        result_complex.im.abs() < 1e-10,
        "Wright Bessel complex imaginary part test failed"
    );

    // Test prolate spheroidal characteristic value
    let pro_result = pro_cv(0, 0, 0.0)?;
    println!("Prolate spheroidal CV(0, 0, 0.0) = {}", pro_result);
    assert!((pro_result - 0.0).abs() < 1e-10, "Prolate CV test failed");

    // Test oblate spheroidal characteristic value
    let obl_result = obl_cv(0, 1, 0.0)?;
    println!("Oblate spheroidal CV(0, 1, 0.0) = {}", obl_result);
    assert!((obl_result - 2.0).abs() < 1e-10, "Oblate CV test failed");

    // Test prolate spheroidal angular function
    let ang_result = pro_ang1(0, 0, 0.0, 0.5)?;
    println!(
        "Prolate spheroidal angular(0, 0, 0.0, 0.5) = {:?}",
        ang_result
    );
    assert!(
        (ang_result.0 - 1.0).abs() < 1e-10,
        "Prolate angular function test failed"
    );

    // Test Wright Bessel with non-zero values
    let result2 = wright_bessel(1.0, 2.0, 0.0)?;
    println!("Wright Bessel(1.0, 2.0, 0.0) = {}", result2);
    assert!(
        (result2 - 1.0).abs() < 1e-10,
        "Wright Bessel(1.0, 2.0, 0.0) test failed"
    );

    // Test spheroidal with small c value
    let pro_result_small_c = pro_cv(0, 1, 0.1)?;
    println!("Prolate spheroidal CV(0, 1, 0.1) = {}", pro_result_small_c);
    assert!(
        pro_result_small_c > 2.0,
        "Prolate CV with small c should be > 2.0"
    );

    println!("All tests passed successfully!");
    Ok(())
}
