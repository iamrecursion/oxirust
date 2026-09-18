//! Example demonstrating numerical stability improvements
//!
//! This example shows how the stable algorithms avoid common numerical
//! pitfalls like catastrophic cancellation, overflow, and loss of precision.

use scirs2_core::ndarray_ext::{array, Array1, Array2};
use scirs2_core::numeric::stability::*;
use scirs2_core::numeric::stable_algorithms::*;
use scirs2_core::CoreResult;

#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("=== Numerical Stability Improvements Example ===\n");

    // Example 1: Stable summation
    println!("1. Stable Summation:");
    demo_stable_summation()?;

    // Example 2: Stable variance calculation
    println!("\n2. Stable Variance Calculation:");
    demo_stable_variance()?;

    // Example 3: Log-sum-exp trick
    println!("\n3. Log-Sum-Exp for Overflow Prevention:");
    demolog_sum_exp()?;

    // Example 4: Stable matrix operations
    println!("\n4. Stable Matrix Operations:");
    demo_stablematrix_ops()?;

    // Example 5: Iterative methods with stability
    println!("\n5. Stable Iterative Methods:");
    demo_iterative_methods()?;

    // Example 6: Numerical differentiation
    println!("\n6. Stable Numerical Differentiation:");
    demo_numerical_differentiation()?;

    // Example 7: Condition number and numerical stability
    println!("\n7. Condition Number Analysis:");
    demo_condition_number()?;

    // Example 8: Stable special functions
    println!("\n8. Stable Special Function Evaluation:");
    demo_stable_special_functions()?;

    Ok(())
}

#[allow(dead_code)]
fn demo_stable_summation() -> CoreResult<()> {
    // Case 1: Adding many small numbers to a large number
    let large = 1e10_f64;
    let small = 1e-5_f64;
    let n = 1_000_000;

    // Naive summation
    let mut naive_sum = large;
    for _ in 0..n {
        naive_sum += small;
    }

    // Kahan summation
    let mut kahan = KahanSum::new();
    kahan.add(large);
    for _ in 0..n {
        kahan.add(small);
    }

    // Neumaier summation
    let mut values = vec![large];
    values.extend(vec![small; n]);
    let neumaier_result = neumaier_sum(&values);

    println!("Adding {n} small values ({small}) to a large value ({large}):");
    println!("Expected result: {}", large + (n as f64) * small);
    println!("Naive sum:       {naive_sum}");
    println!("Kahan sum:       {}", kahan.sum());
    println!("Neumaier sum:    {neumaier_result}");
    println!(
        "Error (naive):   {}",
        (naive_sum - (large + (n as f64) * small)).abs()
    );
    println!(
        "Error (Kahan):   {}",
        (kahan.sum() - (large + (n as f64) * small)).abs()
    );
    println!(
        "Error (Neumaier): {}",
        (neumaier_result - (large + (n as f64) * small)).abs()
    );

    // Case 2: Alternating series that should cancel
    println!("\nAlternating series (1 - 1 + 1 - 1 + ...):");
    let alternating: Vec<f64> = (0..1000000)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let naive: f64 = alternating.iter().sum();
    let stable = pairwise_sum(&alternating);

    println!("Expected: 0.0");
    println!("Naive sum: {naive}");
    println!("Pairwise sum: {stable}");

    Ok(())
}

#[allow(dead_code)]
fn demo_stable_variance() -> CoreResult<()> {
    // Dataset with large mean and small variance
    let mean = 1e8;
    let std = 1.0;
    let n = 10000;

    // Generate data points
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // Simple synthetic data
        let value = mean + std * ((i as f64 / n as f64) - 0.5);
        data.push(value);
    }

    // Naive variance (single pass)
    let naive_mean: f64 = data.iter().sum::<f64>() / n as f64;
    let naive_var: f64 =
        data.iter().map(|&x| (x - naive_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    // Welford's online algorithm
    let mut welford = WelfordVariance::new();
    for &value in &data {
        welford.add(value);
    }

    // Two-pass stable algorithm
    let stable_var = stable_variance(&data, 1)?;

    println!("Dataset: {n} points with large mean ({mean}) and small variance");
    println!("Expected variance: ~{}", std * std);
    println!("Naive single-pass variance: {naive_var}");
    println!(
        "Welford's algorithm variance: {}",
        welford.variance().expect("Operation failed")
    );
    println!("Two-pass stable variance: {stable_var}");

    Ok(())
}

#[allow(dead_code)]
fn demolog_sum_exp() -> CoreResult<()> {
    // Case 1: Large values that would overflow
    let large_values: Vec<f64> = vec![700.0, 701.0, 702.0, 703.0, 704.0];

    println!("Large values that would overflow with naive exp:");
    println!("Values: {large_values:?}");

    // Naive computation would overflow
    let mut naive_overflow = false;
    let mut naive_sum: f64 = 0.0;
    for &v in &large_values {
        let exp_v = v.exp();
        if exp_v.is_infinite() {
            naive_overflow = true;
            break;
        }
        naive_sum += exp_v;
    }

    if naive_overflow {
        println!("Naive computation: OVERFLOW!");
    } else {
        println!("Naive log(sum(exp)): {}", naive_sum.ln());
    }

    let stable_result = log_sum_exp(&large_values);
    println!("Stable log-sum-exp: {stable_result}");

    // Case 2: Softmax with extreme values
    println!("\nSoftmax with extreme values:");
    let extreme_values = vec![1000.0, 0.0, -1000.0];
    let softmax_result = stable_softmax(&extreme_values);

    println!("Input: {extreme_values:?}");
    println!("Stable softmax: {softmax_result:?}");
    println!(
        "Sum of probabilities: {}",
        softmax_result.iter().sum::<f64>()
    );

    Ok(())
}

#[allow(dead_code)]
fn demo_stablematrix_ops() -> CoreResult<()> {
    // Ill-conditioned matrix (Hilbert matrix)
    let n = 5;
    let mut hilbert = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            hilbert[[i, j]] = 1.0 / ((i + j + 2) as f64);
        }
    }

    println!("Hilbert matrix (ill-conditioned):");
    println!("{hilbert:?}");

    // Test with a simple right-hand side
    let b = Array1::ones(n);

    // Gaussian elimination with partial pivoting
    match gaussian_elimination_stable(&hilbert.view(), &b.view()) {
        Ok(x) => {
            println!("\nGaussian elimination solution: {x:?}");

            // Check residual
            let residual = &b - &hilbert.dot(&x);
            let residual_norm = stable_norm_2(&residual.to_vec());
            println!("Residual norm: {residual_norm}");
        }
        Err(e) => {
            println!("\nGaussian elimination failed: {e}");
        }
    }

    // QR decomposition
    let (q, r) = qr_decomposition_stable(&hilbert.view())?;
    println!("\nQR decomposition completed");
    println!("Q orthogonality check (Q^T * Q):");
    let qtq = q.t().dot(&q);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let error = (qtq[[i, j]] - expected).abs();
            if error > 1e-10 {
                println!("  Q^T*Q[{},{}] = {} (error: {})", i, j, qtq[[i, j]], error);
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_iterative_methods() -> CoreResult<()> {
    // Create a symmetric positive definite matrix
    let n = 10;
    let mut a = Array2::zeros((n, n));

    // Tridiagonal matrix (well-conditioned)
    for i in 0..n {
        a[[i, i]] = 4.0;
        if i > 0 {
            a[[i, i.saturating_sub(1)]] = -1.0;
            a[[i - 1, i]] = -1.0;
        }
    }

    let b = Array1::ones(n);

    // Configure iterative solver
    let config = IterativeConfig {
        max_iterations: 100,
        abs_tolerance: 1e-10,
        reltolerance: 1e-8,
        adaptive_tolerance: true,
    };

    // Conjugate gradient
    println!("Conjugate Gradient Method:");
    let cg_result = conjugate_gradient(&a.view(), &b.view(), None, &config)?;

    println!("Converged: {}", cg_result.converged);
    println!("Iterations: {}", cg_result.iterations);
    println!("Final residual: {}", cg_result.residual);

    if let Some(history) = &cg_result.history {
        println!("\nConvergence history (first 10 iterations):");
        for (i, &res) in history.iter().take(10).enumerate() {
            println!("  Iteration {}: residual = {}", i + 1, res);
        }
    }

    // GMRES
    println!("\nGMRES Method:");
    let gmres_result = gmres(&a.view(), &b.view(), None, 5, &config)?;

    println!("Converged: {}", gmres_result.converged);
    println!("Iterations: {}", gmres_result.iterations);
    println!("Final residual: {}", gmres_result.residual);

    Ok(())
}

#[allow(dead_code)]
fn demo_numerical_differentiation() -> CoreResult<()> {
    // Function with known derivative: f(x) = sin(x) * exp(x)
    // f'(x) = cos(x) * exp(x) + sin(x) * exp(x)
    let f = |x: f64| x.sin() * x.exp();
    let f_prime = |x: f64| x.cos() * x.exp() + x.sin() * x.exp();

    let x = 1.0;
    let true_derivative = f_prime(x);

    println!("Function: f(x) = sin(x) * exp(x)");
    println!("Point: x = {x}");
    println!("True derivative: {true_derivative}");

    // Compare different orders of Richardson extrapolation
    println!("\nRichardson extrapolation with different orders:");
    for order in 1..=5 {
        let h = 0.1;
        let approx = richardson_derivative(f, x, h, order)?;
        let error = (approx - true_derivative).abs();
        println!("  Order {order}: {approx} (error: {error})");
    }

    // Demonstrate adaptive integration
    println!("\nAdaptive Simpson's integration:");
    let g = |x: f64| x.sin();
    let a = 0.0;
    let b = std::f64::consts::PI;

    let tolerances = [1e-3, 1e-6, 1e-9, 1e-12];
    let true_integral = 2.0; // integral of sin from 0 to π

    for &tol in &tolerances {
        let result = adaptive_simpson(g, a, b, tol, 20)?;
        let error = (result - true_integral).abs();
        println!("  Tolerance {tol}: result = {result} (error: {error})");
    }

    Ok(())
}

#[allow(dead_code)]
fn demo_condition_number() -> CoreResult<()> {
    // Well-conditioned matrix
    let well_conditioned = array![[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];

    // Ill-conditioned matrix
    let ill_conditioned = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0001, 1.0002],
        [1.0, 1.0002, 1.0004]
    ];

    println!("Matrix condition number estimation:");

    let cond_well = condition_number_estimate(&well_conditioned.view())?;
    println!("\nWell-conditioned matrix:");
    println!("{well_conditioned:?}");
    println!("Estimated condition number: {cond_well}");

    let cond_ill = condition_number_estimate(&ill_conditioned.view())?;
    println!("\nIll-conditioned matrix:");
    println!("{ill_conditioned:?}");
    println!("Estimated condition number: {cond_ill}");

    println!("\nInterpretation:");
    println!("- Condition number close to 1: well-conditioned");
    println!("- Large condition number: ill-conditioned (sensitive to errors)");

    Ok(())
}

#[allow(dead_code)]
fn demo_stable_special_functions() -> CoreResult<()> {
    // Demonstrate stable computation of special functions

    // log(1 + x) for small x
    println!("Stable log(1 + x) computation:");
    let small_values = [1e-15, 1e-10, 1e-5, 0.01, 0.1];

    for &x in &small_values {
        let naive = (1.0_f64 + x).ln();
        let stable = log1p_stable(x);
        let true_value = x - x * x / 2.0 + x * x * x / 3.0; // Taylor series

        println!("\nx = {x}:");
        println!("  Naive log(1+x): {naive}");
        println!("  Stable log1p:   {stable}");
        println!("  Taylor approx:  {true_value}");
        println!(
            "  Relative error (naive): {}",
            ((naive - true_value) / true_value).abs()
        );
        println!(
            "  Relative error (stable): {}",
            ((stable - true_value) / true_value).abs()
        );
    }

    // Hypot for extreme values
    println!("\n\nStable hypot(x, y) = sqrt(x² + y²):");
    let test_cases = [(3.0, 4.0), (1e200, 1e200), (1e-200, 1e-200), (1e150, 1e100)];

    for (x, y) in test_cases {
        let stable = hypot_stable(x, y);
        println!("\nhypot({x}, {y}) = {stable}");

        // Check if naive computation would work
        let naive_sq: f64 = x * x + y * y;
        if naive_sq.is_infinite() {
            println!("  Naive x²+y² would overflow!");
        } else if naive_sq == 0.0 && (x != 0.0 || y != 0.0) {
            println!("  Naive x²+y² would underflow!");
        }
    }

    // Cross entropy with extreme probabilities
    println!("\n\nStable cross-entropy loss:");
    let predictions = vec![0.999999, 0.000001, 0.5];
    let targets = vec![1.0, 0.0, 1.0];

    let loss = cross_entropy_stable(&predictions, &targets)?;
    println!("Predictions: {predictions:?}");
    println!("Targets: {targets:?}");
    println!("Cross-entropy loss: {loss}");

    Ok(())
}
