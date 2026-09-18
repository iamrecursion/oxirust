use scirs2_special::{ai, aip, bi, bip};

#[allow(dead_code)]
fn main() {
    println!("Example of Airy functions");
    println!("-----------------------");

    // Define some values to evaluate the functions
    let points = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];

    println!("| x | Ai(x) | Ai'(x) | Bi(x) | Bi'(x) |");
    println!("|---|-------|--------|-------|--------|");

    for x in points {
        // Evaluate Airy functions
        let ai_x = ai(x);
        let aip_x = aip(x);
        let bi_x = bi(x);
        let bip_x = bip(x);

        // Print results
        println!(
            "| {:.1} | {:.6} | {:.6} | {:.6} | {:.6} |",
            x, ai_x, aip_x, bi_x, bip_x
        );
    }

    // Verify the relation: Ai(x)·Bi'(x) - Ai'(x)·Bi(x) = 1/π
    let x = 2.0;
    let wroskian = ai(x) * bip(x) - aip(x) * bi(x);
    println!("\nWronskian at x = 2.0: {:.10}", wroskian);
    println!("1/π: {:.10}", 1.0 / std::f64::consts::PI);
    println!(
        "Difference: {:.10}",
        (wroskian - 1.0 / std::f64::consts::PI).abs()
    );
}
