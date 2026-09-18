use scirs2_special::{elliptic_e, elliptic_e_inc, elliptic_f, elliptic_k, elliptic_pi};
use scirs2_special::{jacobi_cn, jacobi_dn, jacobi_sn};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() {
    println!("=== Elliptic Integrals and Functions Example ===\n");

    // Parameters
    let m_values = [0.0, 0.25, 0.5, 0.75, 0.99];
    let phi = PI / 3.0; // 60 degrees
    let u = 1.0;

    println!("Complete Elliptic Integrals of the First Kind (K):");
    println!("--------------------------------------------------");
    println!("m\t\tK(m)");
    println!("--------------------------------------------------");
    for m in m_values {
        println!("{:.2}\t\t{:.8}", m, elliptic_k(m));
    }
    println!();

    println!("Complete Elliptic Integrals of the Second Kind (E):");
    println!("--------------------------------------------------");
    println!("m\t\tE(m)");
    println!("--------------------------------------------------");
    for m in m_values {
        println!("{:.2}\t\t{:.8}", m, elliptic_e(m));
    }
    println!();

    println!("Incomplete Elliptic Integrals of the First Kind (F):");
    println!("--------------------------------------------------");
    println!("m\t\tF(φ=π/3, m)");
    println!("--------------------------------------------------");
    for m in m_values {
        println!("{:.2}\t\t{:.8}", m, elliptic_f(phi, m));
    }
    println!();

    println!("Incomplete Elliptic Integrals of the Second Kind (E):");
    println!("--------------------------------------------------");
    println!("m\t\tE(φ=π/3, m)");
    println!("--------------------------------------------------");
    for m in m_values {
        println!("{:.2}\t\t{:.8}", m, elliptic_e_inc(phi, m));
    }
    println!();

    // Elliptic integrals of the third kind with different characteristic values
    let n_values = [0.0, 0.25, 0.5];
    println!("Incomplete Elliptic Integrals of the Third Kind (Π):");
    println!("--------------------------------------------------");
    println!("n\t\tm\t\tΠ(n, φ=π/4, m)");
    println!("--------------------------------------------------");
    for n in n_values {
        for m in &[0.0, 0.5] {
            println!(
                "{:.2}\t\t{:.2}\t\t{:.8}",
                n,
                m,
                elliptic_pi(n, PI / 4.0, *m)
            );
        }
    }
    println!();

    // Jacobi elliptic functions for various m values
    println!("Jacobi Elliptic Functions at u = {}:", u);
    println!("--------------------------------------------------");
    println!("m\t\tsn(u,m)\t\tcn(u,m)\t\tdn(u,m)");
    println!("--------------------------------------------------");
    for m in m_values {
        let sn = jacobi_sn(u, m);
        let cn = jacobi_cn(u, m);
        let dn = jacobi_dn(u, m);
        println!("{:.2}\t\t{:.8}\t{:.8}\t{:.8}", m, sn, cn, dn);

        // Verify Jacobi identities
        println!("    Identity 1: sn²+cn² = {:.10}", sn * sn + cn * cn);
        println!("    Identity 2: m·sn²+dn² = {:.10}", m * sn * sn + dn * dn);
    }

    // Relationship between elliptic integrals and Jacobi elliptic functions
    println!("\nRelationship between Elliptic Integrals and Jacobi Elliptic Functions:");
    println!("--------------------------------------------------");
    let m = 0.5;
    let amplitude = PI / 4.0;

    // For a given amplitude and parameter m, compute F(φ|m)
    let f_value = elliptic_f(amplitude, m);

    // The relationship: If u = F(φ|m), then sn(u,m) = sin(φ)
    let sn_value = jacobi_sn(f_value, m);
    let sin_value = amplitude.sin();

    println!("F(φ=π/4|m=0.5) = {:.8}", f_value);
    println!("sn(F(φ|m), m) = {:.8}", sn_value);
    println!("sin(φ) = {:.8}", sin_value);
    println!("Difference: {:.10}", (sn_value - sin_value).abs());
}
