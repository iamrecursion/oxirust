use scirs2_special::{
    ai, aip, betainc, bi, bip, erf, erfc, erfcinv, erfinv, hurwitz_zeta, k0, k1, mathieu_a,
    mathieu_b, mathieu_cem, mathieu_even_coef, mathieu_odd_coef, mathieu_sem, y0, y1, zeta, zetac,
};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() {
    let x1 = 3.957678419314858;
    let x2 = 5.429;
    let x3 = 1.0;
    let y = 0.5f64;

    println!("y0({}) = {}", x1, y0(x1));
    println!("y1({}) = {}", x2, y1(x2));
    println!("y0({}) = {}", x3, y0(x3));
    println!("y1({}) = {}", x3, y1(x3));
    println!("k0({}) = {}", x3, k0(x3));
    println!("k1({}) = {}", x3, k1(x3));

    println!("\nTesting erf functions:");
    println!("erf(0) = {}", erf(0.0));
    println!("erfc(0) = {}", erfc(0.0));

    let x_inv = erfinv(y);
    println!("erfinv({}) = {}", y, x_inv);
    println!("erf(erfinv({})) = {}", y, erf(x_inv));

    let x_cinv = erfcinv(y);
    println!("erfcinv({}) = {}", y, x_cinv);
    println!("erfc(erfcinv({})) = {}", y, erfc(x_cinv));

    // Test beta inc
    let x_beta = 0.5f64;
    let a = 2.0f64;
    let b = 3.0f64;
    let inc_beta = betainc(x_beta, a, b).expect("Operation failed");
    println!("\nBeta inc function:");
    println!("betainc({}, {}, {}) = {}", x_beta, a, b, inc_beta);

    // Test zeta functions
    println!("\nZeta functions:");
    let s2 = 2.0f64;
    let s4 = 4.0f64;
    println!("zeta({}) = {}", s2, zeta(s2).expect("Operation failed"));
    println!("zeta({}) = {}", s4, zeta(s4).expect("Operation failed"));
    println!("zetac({}) = {}", s2, zetac(s2).expect("Operation failed"));
    println!(
        "hurwitz_zeta({}, 1.0) = {}",
        s2,
        hurwitz_zeta(s2, 1.0).expect("Operation failed")
    );

    // Test Airy functions
    println!("\nAiry functions:");
    println!("ai(0.0) = {}", ai(0.0));
    println!("ai(1.0) = {}", ai(1.0));
    println!("ai(-1.0) = {}", ai(-1.0));
    println!("aip(0.0) = {}", aip(0.0));
    println!("bi(0.0) = {}", bi(0.0));
    println!("bi(1.0) = {}", bi(1.0));
    println!("bip(0.0) = {}", bip(0.0));

    // Test Mathieu functions
    println!("\nMathieu functions:");

    // Test characteristic values
    let q_values = [0.0, 0.1, 1.0, 5.0];

    println!("\nCharacteristic values (a_m):");
    for &q in &q_values {
        println!("  q = {}:", q);
        for m in 0..5 {
            let a = mathieu_a(m, q).expect("Operation failed");
            println!("    mathieu_a({}, {}) = {}", m, q, a);
        }
    }

    println!("\nCharacteristic values (b_m):");
    for &q in &q_values {
        println!("  q = {}:", q);
        for m in 1..5 {
            // Note: m=0 is not valid for odd Mathieu functions
            let b = mathieu_b(m, q).expect("Operation failed");
            println!("    mathieu_b({}, {}) = {}", m, q, b);
        }
    }

    // Test Fourier coefficients
    println!("\nFourier coefficients:");
    let q_test = 1.0;

    println!("  Even coefficients (m=0, q={})", q_test);
    let even_coeffs = mathieu_even_coef(0, q_test).expect("Operation failed");
    for (i, &coef) in even_coeffs.iter().enumerate().take(5) {
        println!("    A_0^{} = {}", 2 * i, coef);
    }

    println!("  Odd coefficients (m=1, q={})", q_test);
    let odd_coeffs = mathieu_odd_coef(1, q_test).expect("Operation failed");
    for (i, &coef) in odd_coeffs.iter().enumerate().take(5) {
        println!("    B_1^{} = {}", 2 * i + 1, coef);
    }

    // Test function values
    let x_values = [0.0, PI / 4.0, PI / 2.0];

    println!("\nMathieu function values:");
    for &x in &x_values {
        println!("  x = {}:", x);

        // Even Mathieu functions
        let (ce0, dce0) = mathieu_cem(0, q_test, x).expect("Operation failed");
        println!("    ce_0({}, {}) = {}", q_test, x, ce0);
        println!("    ce_0'({}, {}) = {}", q_test, x, dce0);

        let (ce1, dce1) = mathieu_cem(1, q_test, x).expect("Operation failed");
        println!("    ce_1({}, {}) = {}", q_test, x, ce1);
        println!("    ce_1'({}, {}) = {}", q_test, x, dce1);

        // Odd Mathieu functions
        let (se1, dse1) = mathieu_sem(1, q_test, x).expect("Operation failed");
        println!("    se_1({}, {}) = {}", q_test, x, se1);
        println!("    se_1'({}, {}) = {}", q_test, x, dse1);

        let (se2, dse2) = mathieu_sem(2, q_test, x).expect("Operation failed");
        println!("    se_2({}, {}) = {}", q_test, x, se2);
        println!("    se_2'({}, {}) = {}", q_test, x, dse2);
    }
}
