//! Demo of enhanced linear operators with SIMD and parallel acceleration
//!
//! This example demonstrates the new enhanced linear operators that provide
//! performance optimizations using SIMD acceleration and parallel processing.

use scirs2_sparse::{
    convolution_operator, enhanced_add, enhanced_diagonal, enhanced_scale,
    finite_difference_operator, BoundaryCondition, ConvolutionMode, EnhancedDiagonalOperator,
    EnhancedOperatorOptions, LinearOperator,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Enhanced Linear Operators Demo");
    println!("==============================");

    // Example 1: Enhanced diagonal operator
    println!("\n1. Enhanced Diagonal Operator:");
    let diagonal = vec![2.0, 3.0, 4.0, 5.0];
    let diag_op = enhanced_diagonal(diagonal);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = diag_op.matvec(&x)?;
    println!("   Input: {:?}", x);
    println!("   Diagonal: [2, 3, 4, 5]");
    println!("   Result: {:?}", result);
    println!("   Expected: [2, 6, 12, 20]");

    // Example 2: Convolution operator
    println!("\n2. Convolution Operator:");
    let kernel = vec![1.0, 2.0, 1.0]; // Simple smoothing kernel
    let conv_op = convolution_operator(kernel, 5, ConvolutionMode::Same);
    let signal = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let filtered = conv_op.matvec(&signal)?;
    println!("   Signal: {:?}", signal);
    println!("   Kernel: [1, 2, 1]");
    println!("   Filtered: {:?}", filtered);

    // Example 3: Finite difference operator
    println!("\n3. Finite Difference Operator (1st derivative):");
    let fd_op = finite_difference_operator(5, 1, 1.0, BoundaryCondition::Dirichlet);
    let func_values = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // f(x) = x^2 at x = 0,1,2,3,4
    let derivative = fd_op.matvec(&func_values)?;
    println!("   Function values: {:?}", func_values);
    println!("   Derivative approximation: {:?}", derivative);
    println!("   Expected ~[0, 2, 4, 6, 8] (derivative of x^2 is 2x)");

    // Example 4: Operator composition
    println!("\n4. Enhanced Operator Composition:");
    let diag1 = enhanced_diagonal(vec![1.0, 2.0, 3.0]);
    let diag2 = enhanced_diagonal(vec![2.0, 1.0, 1.0]);

    // Create sum of two diagonal operators
    let sum_op = enhanced_add(diag1, diag2)?;
    let x = vec![1.0, 1.0, 1.0];
    let sum_result = sum_op.matvec(&x)?;
    println!("   Input: {:?}", x);
    println!("   Diagonal 1: [1, 2, 3]");
    println!("   Diagonal 2: [2, 1, 1]");
    println!("   Sum result: {:?}", sum_result);
    println!("   Expected: [3, 3, 4]");

    // Example 5: Scaled operator
    println!("\n5. Enhanced Scaled Operator:");
    let identity_diag = enhanced_diagonal(vec![1.0, 1.0, 1.0, 1.0]);
    let scaled_op = enhanced_scale(5.0, identity_diag);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let scaled_result = scaled_op.matvec(&x)?;
    println!("   Input: {:?}", x);
    println!("   Scale factor: 5.0");
    println!("   Result: {:?}", scaled_result);
    println!("   Expected: [5, 10, 15, 20]");

    // Example 6: Performance configuration
    println!("\n6. Custom Performance Options:");
    let custom_options = EnhancedOperatorOptions {
        use_parallel: true,
        parallel_threshold: 1000, // Lower threshold for demo
        use_simd: true,
        simd_threshold: 16, // Lower threshold for demo
        chunk_size: 256,
    };

    let large_diagonal: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    let enhanced_diag = EnhancedDiagonalOperator::with_options(large_diagonal, custom_options);
    let large_input = vec![1.0; 1000];
    let large_result = enhanced_diag.matvec(&large_input)?;

    println!("   Large vector operation completed");
    println!("   Input size: {}", large_input.len());
    println!("   Result size: {}", large_result.len());
    println!("   First few results: {:?}", &large_result[0..5]);
    println!("   Last few results: {:?}", &large_result[995..1000]);

    println!("\n7. Performance Features Summary:");
    println!("   ✓ SIMD acceleration for element-wise operations");
    println!("   ✓ Parallel processing for large vectors");
    println!("   ✓ Configurable performance thresholds");
    println!("   ✓ Integration with scirs2-core infrastructure");
    println!("   ✓ Matrix-free operator implementations");
    println!("   ✓ Specialized operators (convolution, finite difference)");
    println!("   ✓ Operator composition and algebraic operations");

    println!("\nDemo completed successfully!");
    Ok(())
}
