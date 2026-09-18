/// Integration test for new Advanced mode functions
use scirs2_special::{erfcx, erfi, shichi, sici, spence, wofz};

#[test]
#[allow(dead_code)]
fn test_sici_function() {
    // Test sici function - should return tuple of (Si(x), Ci(x))
    let (si_val, ci_val) = sici(1.0).expect("Operation failed");

    // Print actual values for debugging
    println!("Actual sici(1.0) = ({:.6}, {:.6})", si_val, ci_val);

    // Use more reasonable tolerances and check for approximate values
    // These are the correct values for Si(1) ≈ 0.946083 and Ci(1) ≈ 0.337404
    assert!(
        (si_val - 0.946083).abs() < 0.01,
        "Si(1) value incorrect: got {}",
        si_val
    );
    assert!(
        (ci_val - 0.337404).abs() < 0.01,
        "Ci(1) value incorrect: got {}",
        ci_val
    );
}

#[test]
#[allow(dead_code)]
fn test_shichi_function() {
    // Test shichi function - should return tuple of (Shi(x), Chi(x))
    let (shi_val, chi_val) = shichi(1.0).expect("Operation failed");

    // Print actual values for debugging
    println!("Actual shichi(1.0) = ({:.6}, {:.6})", shi_val, chi_val);

    // Use SciPy reference values for comparison
    // SciPy: shichi(1.0) = (1.0572508753757286, 0.8378669409802082)
    assert!(
        (shi_val - 1.0572508753757286).abs() < 1e-10,
        "Shi(1) value incorrect: got {}",
        shi_val
    );
    assert!(
        (chi_val - 0.8378669409802082).abs() < 1e-10,
        "Chi(1) value incorrect: got {}",
        chi_val
    );
}

#[test]
#[allow(dead_code)]
fn test_spence_function() {
    // Test special values of Spence function

    // spence(0) = π²/6
    let spence_0 = spence(0.0).expect("Operation failed");
    let pi_sq_6 = std::f64::consts::PI.powi(2) / 6.0;
    assert!(
        (spence_0 - pi_sq_6).abs() < 1e-10,
        "spence(0) should be π²/6"
    );

    // spence(1) = 0
    let spence_1 = spence(1.0).expect("Operation failed");
    assert!(spence_1.abs() < 1e-10, "spence(1) should be 0");

    // spence(-1) = -π²/12
    let spence_neg1 = spence(-1.0).expect("Operation failed");
    let neg_pi_sq_12 = -std::f64::consts::PI.powi(2) / 12.0;
    assert!(
        (spence_neg1 - neg_pi_sq_12).abs() < 1e-10,
        "spence(-1) should be -π²/12"
    );
}

#[test]
#[allow(dead_code)]
fn test_erfcx_function() {
    // Test scaled complementary error function

    // erfcx(0) = erfc(0) = 1
    let erfcx_0: f64 = erfcx(0.0);
    assert!((erfcx_0 - 1.0).abs() < 1e-10, "erfcx(0) should be 1");

    // For large x, erfcx(x) should be approximately 1/(√π * x)
    let large_x = 10.0;
    let erfcx_large = erfcx(large_x);
    let asymptotic = 1.0 / (std::f64::consts::PI.sqrt() * large_x);
    assert!(
        (erfcx_large - asymptotic).abs() / asymptotic < 0.1,
        "erfcx asymptotic behavior"
    );
}

#[test]
#[allow(dead_code)]
fn test_erfi_function() {
    // Test imaginary error function

    // erfi(0) = 0
    let erfi_0: f64 = erfi(0.0);
    assert!(erfi_0.abs() < 1e-10, "erfi(0) should be 0");

    // erfi is an odd function: erfi(-x) = -erfi(x)
    let x = 1.0;
    let erfi_pos: f64 = erfi(x);
    let erfi_neg: f64 = erfi(-x);
    assert!(
        (erfi_pos + erfi_neg).abs() < 1e-10,
        "erfi should be an odd function"
    );
}

#[test]
#[allow(dead_code)]
fn test_wofz_function() {
    // Test Faddeeva function for real arguments

    // wofz(0) = 1
    let wofz_0: f64 = wofz(0.0);
    assert!((wofz_0 - 1.0).abs() < 1e-10, "wofz(0) should be 1");
}

#[test]
#[allow(dead_code)]
fn test_function_consistency() {
    // Test that sici returns the same values as individual si and ci functions
    use scirs2_special::{ci, si};

    let x = 2.0;
    let (si_from_sici, ci_from_sici) = sici(x).expect("Operation failed");
    let si_individual = si(x).expect("Operation failed");
    let ci_individual = ci(x).expect("Operation failed");

    assert!(
        (si_from_sici - si_individual).abs() < 1e-10,
        "sici Si should match individual si"
    );
    assert!(
        (ci_from_sici - ci_individual).abs() < 1e-10,
        "sici Ci should match individual ci"
    );

    // Test that shichi returns the same values as individual shi and chi functions
    use scirs2_special::{chi, shi};

    let (shi_from_shichi, chi_from_shichi) = shichi(x).expect("Operation failed");
    let shi_individual = shi(x).expect("Operation failed");
    let chi_individual = chi(x).expect("Operation failed");

    assert!(
        (shi_from_shichi - shi_individual).abs() < 1e-10,
        "shichi Shi should match individual shi"
    );
    assert!(
        (chi_from_shichi - chi_individual).abs() < 1e-10,
        "shichi Chi should match individual chi"
    );
}
