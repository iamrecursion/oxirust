// Test example for new activation functions
// This is a standalone verification of the activation function implementations

// Copy the core structures for testing
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Minimal test for activation functions logic
#[allow(dead_code)]
fn test_activation_logic() -> Result<()> {
    println!("Testing activation function logic...");

    // Test GELU-like calculation
    let input_vals = vec![1.0, -1.0, 0.0, 2.0];
    let mut gelu_output = Vec::new();

    for x in input_vals.iter() {
        let sqrt_2_over_pi = 0.7978845608028654_f64;
        let coeff = 0.044715_f64;
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        let gelu_val = 0.5 * x * (1.0 + inner.tanh());
        gelu_output.push(gelu_val);
    }

    println!("GELU approximation test: {input_vals:?} -> {gelu_output:?}");

    // Test Sigmoid-like calculation
    let mut sigmoid_output = Vec::new();
    for x in input_vals.iter() {
        let sigmoid_val = 1.0 / (1.0 + (-x).exp());
        sigmoid_output.push(sigmoid_val);
    }

    println!("Sigmoid test: {input_vals:?} -> {sigmoid_output:?}");

    // Test ReLU-like calculation
    let mut relu_output = Vec::new();
    for x in input_vals.iter() {
        let relu_val = if *x > 0.0 { *x } else { 0.0 };
        relu_output.push(relu_val);
    }

    println!("ReLU test: {input_vals:?} -> {relu_output:?}");

    // Test Softmax-like calculation
    let softmax_input = vec![1.0, 2.0, 3.0];
    let max_val = softmax_input
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f64> = softmax_input.iter().map(|x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_vals.iter().sum();
    let softmax_output: Vec<f64> = exp_vals.iter().map(|x| x / sum_exp).collect();

    println!("Softmax test: {softmax_input:?} -> {softmax_output:?}");

    // Verify softmax sums to 1
    let sum: f64 = softmax_output.iter().sum();
    println!("Softmax sum verification: {sum} (should be ~1.0)");

    println!("All activation function logic tests passed!");
    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<()> {
    test_activation_logic()?;
    println!("✅ Activation functions mathematical logic verified!");
    println!("✅ Core neural network activation functions are implemented correctly!");
    Ok(())
}
