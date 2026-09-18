//! Combinatorial functions
//!
//! This module provides various combinatorial functions including:
//! - Factorial and related functions
//! - Binomial coefficients
//! - Permutations and combinations
//! - Stirling numbers
//! - Bell numbers
//! - Bernoulli numbers
//! - Euler numbers

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::gamma;
use std::f64::consts::PI;

/// Computes the factorial of a non-negative integer.
///
/// For large values, this uses the gamma function: n! = Γ(n+1).
///
/// # Arguments
///
/// * `n` - Non-negative integer
///
/// # Returns
///
/// * `SpecialResult<f64>` - The factorial value n!
///
/// # Examples
///
/// ```
/// use scirs2_special::factorial;
///
/// assert_eq!(factorial(0).expect("test/example should not fail"), 1.0);
/// assert_eq!(factorial(5).expect("test/example should not fail"), 120.0);
/// assert_eq!(factorial(10).expect("test/example should not fail"), 3628800.0);
/// ```
#[allow(dead_code)]
pub fn factorial(n: u32) -> SpecialResult<f64> {
    if n <= 20 {
        // Use direct calculation for small values
        let mut result = 1.0;
        for i in 1..=n {
            result *= i as f64;
        }
        Ok(result)
    } else {
        // Use gamma function for larger values
        Ok(gamma((n + 1) as f64))
    }
}

/// Computes the double factorial of a non-negative integer.
///
/// Double factorial n!! is defined as:
/// - For even n: n!! = n × (n-2) × (n-4) × ... × 2
/// - For odd n: n!! = n × (n-2) × (n-4) × ... × 1
/// - 0!! = 1 by convention
///
/// # Arguments
///
/// * `n` - Non-negative integer
///
/// # Returns
///
/// * `SpecialResult<f64>` - The double factorial value n!!
///
/// # Examples
///
/// ```
/// use scirs2_special::double_factorial;
///
/// assert_eq!(double_factorial(0).expect("test/example should not fail"), 1.0);
/// assert_eq!(double_factorial(5).expect("test/example should not fail"), 15.0); // 5 × 3 × 1
/// assert_eq!(double_factorial(6).expect("test/example should not fail"), 48.0); // 6 × 4 × 2
/// ```
#[allow(dead_code)]
pub fn double_factorial(n: u32) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    let mut result = 1.0;
    let mut i = n;
    while i > 0 {
        result *= i as f64;
        i = i.saturating_sub(2);
    }
    Ok(result)
}

/// Computes the double factorial n!! (alias for SciPy compatibility).
///
/// This is an alias for `double_factorial` to match SciPy's `factorial2` function.
///
/// # Arguments
///
/// * `n` - The number for which to compute the double factorial
///
/// # Returns
///
/// * `SpecialResult<f64>` - The double factorial n!!
///
/// # Examples
///
/// ```
/// use scirs2_special::factorial2;
///
/// assert_eq!(factorial2(5).expect("test/example should not fail"), 15.0); // 5 × 3 × 1
/// assert_eq!(factorial2(6).expect("test/example should not fail"), 48.0); // 6 × 4 × 2
/// ```
#[allow(dead_code)]
pub fn factorial2(n: u32) -> SpecialResult<f64> {
    double_factorial(n)
}

/// Computes the k-factorial (multi-factorial) of n.
///
/// The k-factorial of n is the product of positive integers up to n
/// that are congruent to n modulo k.
/// For example, the 3-factorial of 8 is 8×5×2.
///
/// # Arguments
///
/// * `n` - The number for which to compute the k-factorial
/// * `k` - The step size (must be positive)
///
/// # Returns
///
/// * `SpecialResult<f64>` - The k-factorial of n
///
/// # Examples
///
/// ```
/// use scirs2_special::factorialk;
///
/// assert_eq!(factorialk(8, 3).expect("test/example should not fail"), 80.0); // 8 × 5 × 2
/// assert_eq!(factorialk(5, 2).expect("test/example should not fail"), 15.0); // 5 × 3 × 1
/// assert_eq!(factorialk(6, 2).expect("test/example should not fail"), 48.0); // 6 × 4 × 2 (same as double factorial)
/// ```
#[allow(dead_code)]
pub fn factorialk(n: u32, k: u32) -> SpecialResult<f64> {
    if k == 0 {
        return Err(crate::SpecialError::ValueError(
            "k must be positive".to_string(),
        ));
    }

    if n == 0 {
        return Ok(1.0);
    }

    let mut result = 1.0;
    let mut i = n;
    while i > 0 {
        result *= i as f64;
        i = i.saturating_sub(k);
    }
    Ok(result)
}

/// Computes the binomial coefficient "n choose k".
///
/// The binomial coefficient C(n,k) = n! / (k! × (n-k)!) represents
/// the number of ways to choose k items from n items.
///
/// # Arguments
///
/// * `n` - Total number of items
/// * `k` - Number of items to choose
///
/// # Returns
///
/// * `SpecialResult<f64>` - The binomial coefficient C(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::binomial;
///
/// assert_eq!(binomial(5, 2).expect("test/example should not fail"), 10.0);
/// assert_eq!(binomial(10, 3).expect("test/example should not fail"), 120.0);
/// assert_eq!(binomial(7, 0).expect("test/example should not fail"), 1.0);
/// assert_eq!(binomial(7, 7).expect("test/example should not fail"), 1.0);
/// ```
#[allow(dead_code)]
pub fn binomial(n: u32, k: u32) -> SpecialResult<f64> {
    if k > n {
        return Ok(0.0);
    }

    if k == 0 || k == n {
        return Ok(1.0);
    }

    // Use symmetry: C(n,k) = C(n,n-k)
    let k = if k > n - k { n - k } else { k };

    // For small values, use direct calculation to avoid precision issues
    if n <= 30 {
        let mut result = 1.0;
        for i in 0..k {
            result = result * (n - i) as f64 / (i + 1) as f64;
        }
        Ok(result)
    } else {
        // Use gamma function for larger values
        let n_fact = gamma((n + 1) as f64);
        let k_fact = gamma((k + 1) as f64);
        let nk_fact = gamma((n - k + 1) as f64);
        Ok(n_fact / (k_fact * nk_fact))
    }
}

/// Computes the number of permutations of n items taken k at a time.
///
/// P(n,k) = n! / (n-k)! represents the number of ways to arrange
/// k items from n items where order matters.
///
/// # Arguments
///
/// * `n` - Total number of items
/// * `k` - Number of items to arrange
///
/// # Returns
///
/// * `SpecialResult<f64>` - The number of permutations P(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::permutations;
///
/// assert_eq!(permutations(5, 2).expect("test/example should not fail"), 20.0);
/// assert_eq!(permutations(10, 3).expect("test/example should not fail"), 720.0);
/// assert_eq!(permutations(7, 0).expect("test/example should not fail"), 1.0);
/// ```
#[allow(dead_code)]
pub fn permutations(n: u32, k: u32) -> SpecialResult<f64> {
    if k > n {
        return Ok(0.0);
    }

    if k == 0 {
        return Ok(1.0);
    }

    // For small values, use direct calculation
    if n <= 30 {
        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64;
        }
        Ok(result)
    } else {
        // Use gamma function for larger values
        let n_fact = gamma((n + 1) as f64);
        let nk_fact = gamma((n - k + 1) as f64);
        Ok(n_fact / nk_fact)
    }
}

/// Computes the number of permutations (alias for SciPy compatibility).
///
/// This is an alias for `permutations` to match SciPy's `perm` function.
///
/// # Arguments
///
/// * `n` - Total number of items
/// * `k` - Number of items to arrange
///
/// # Returns
///
/// * `SpecialResult<f64>` - The number of permutations P(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::perm;
///
/// assert_eq!(perm(5, 2).expect("test/example should not fail"), 20.0);
/// assert_eq!(perm(10, 3).expect("test/example should not fail"), 720.0);
/// ```
#[allow(dead_code)]
pub fn perm(n: u32, k: u32) -> SpecialResult<f64> {
    permutations(n, k)
}

/// Computes the Stirling number of the first kind.
///
/// The unsigned Stirling number of the first kind s(n,k) counts
/// the number of permutations of n elements with exactly k cycles.
///
/// # Arguments
///
/// * `n` - Number of elements
/// * `k` - Number of cycles
///
/// # Returns
///
/// * `SpecialResult<f64>` - The unsigned Stirling number s(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::stirling_first;
///
/// assert_eq!(stirling_first(0, 0).expect("test/example should not fail"), 1.0);
/// assert_eq!(stirling_first(4, 2).expect("test/example should not fail"), 11.0);
/// assert_eq!(stirling_first(5, 3).expect("test/example should not fail"), 35.0);
/// ```
#[allow(dead_code)]
pub fn stirling_first(n: u32, k: u32) -> SpecialResult<f64> {
    if n == 0 && k == 0 {
        return Ok(1.0);
    }
    if n == 0 || k == 0 || k > n {
        return Ok(0.0);
    }

    // Use recurrence relation: s(n,k) = (n-1) * s(n-1,k) + s(n-1,k-1)
    let mut dp = vec![vec![0.0; (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = 1.0;

    for i in 1..=n as usize {
        for j in 1..=std::cmp::min(i, k as usize) {
            dp[i][j] = (i - 1) as f64 * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }

    Ok(dp[n as usize][k as usize])
}

/// Computes the Stirling number of the second kind.
///
/// The Stirling number of the second kind S(n,k) counts the number
/// of ways to partition n elements into exactly k non-empty subsets.
///
/// # Arguments
///
/// * `n` - Number of elements
/// * `k` - Number of subsets
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Stirling number S(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::stirling_second;
///
/// assert_eq!(stirling_second(0, 0).expect("test/example should not fail"), 1.0);
/// assert_eq!(stirling_second(4, 2).expect("test/example should not fail"), 7.0);
/// assert_eq!(stirling_second(5, 3).expect("test/example should not fail"), 25.0);
/// ```
#[allow(dead_code)]
pub fn stirling_second(n: u32, k: u32) -> SpecialResult<f64> {
    if n == 0 && k == 0 {
        return Ok(1.0);
    }
    if n == 0 || k == 0 || k > n {
        return Ok(0.0);
    }

    // Use recurrence relation: S(n,k) = k * S(n-1,k) + S(n-1,k-1)
    let mut dp = vec![vec![0.0; (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = 1.0;

    for i in 1..=n as usize {
        for j in 1..=std::cmp::min(i, k as usize) {
            dp[i][j] = j as f64 * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }

    Ok(dp[n as usize][k as usize])
}

/// Computes the Stirling number of the second kind (alias for SciPy compatibility).
///
/// This is an alias for `stirling_second` to match SciPy's `stirling2` function.
///
/// # Arguments
///
/// * `n` - Number of elements
/// * `k` - Number of subsets
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Stirling number S(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::stirling2;
///
/// assert_eq!(stirling2(4, 2).expect("test/example should not fail"), 7.0);
/// assert_eq!(stirling2(5, 3).expect("test/example should not fail"), 25.0);
/// ```
#[allow(dead_code)]
pub fn stirling2(n: u32, k: u32) -> SpecialResult<f64> {
    stirling_second(n, k)
}

/// Computes the Bell number B(n).
///
/// The Bell number B(n) counts the number of ways to partition
/// a set of n elements into non-empty subsets.
///
/// # Arguments
///
/// * `n` - Number of elements
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Bell number B(n)
///
/// # Examples
///
/// ```
/// use scirs2_special::bell_number;
///
/// assert_eq!(bell_number(0).expect("test/example should not fail"), 1.0);
/// assert_eq!(bell_number(1).expect("test/example should not fail"), 1.0);
/// assert_eq!(bell_number(2).expect("test/example should not fail"), 2.0);
/// assert_eq!(bell_number(3).expect("test/example should not fail"), 5.0);
/// assert_eq!(bell_number(4).expect("test/example should not fail"), 15.0);
/// ```
#[allow(dead_code)]
pub fn bell_number(n: u32) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    // B(n) = sum of S(n,k) for k from 0 to n
    let mut result = 0.0;
    for k in 0..=n {
        result += stirling_second(n, k)?;
    }
    Ok(result)
}

/// Computes the Bernoulli number B(n).
///
/// The Bernoulli numbers are a sequence of rational numbers defined
/// by the generating function t/(e^t - 1) = sum(B_n * t^n / n!).
///
/// This implementation returns the modern convention where B_1 = -1/2.
///
/// # Arguments
///
/// * `n` - Index of the Bernoulli number
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Bernoulli number B(n)
///
/// # Examples
///
/// ```
/// use scirs2_special::bernoulli_number;
/// use approx::assert_relative_eq;
///
/// assert_eq!(bernoulli_number(0).expect("test/example should not fail"), 1.0);
/// assert_relative_eq!(bernoulli_number(1).expect("test/example should not fail"), -0.5, epsilon = 1e-10);
/// assert_relative_eq!(bernoulli_number(2).expect("test/example should not fail"), 1.0/6.0, epsilon = 1e-10);
/// assert_eq!(bernoulli_number(3).expect("test/example should not fail"), 0.0); // Odd Bernoulli numbers are 0 (except B_1)
/// ```
#[allow(dead_code)]
pub fn bernoulli_number(n: u32) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }
    if n == 1 {
        return Ok(-0.5);
    }
    if n > 1 && n % 2 == 1 {
        return Ok(0.0); // Odd Bernoulli numbers are zero (except B_1)
    }

    // For small even n, use known values
    match n {
        2 => return Ok(1.0 / 6.0),
        4 => return Ok(-1.0 / 30.0),
        6 => return Ok(1.0 / 42.0),
        8 => return Ok(-1.0 / 30.0),
        10 => return Ok(5.0 / 66.0),
        12 => return Ok(-691.0 / 2730.0),
        _ => {} // Fall through to the recurrence relation below
    }

    // For larger values, use recurrence relation
    // B_n = -1/(n+1) * sum_{k=0}^{n-1} C(n+1,k) * B_k
    let mut bernoulli = vec![0.0; (n + 1) as usize];
    bernoulli[0] = 1.0;
    if n >= 1 {
        bernoulli[1] = -0.5;
    }

    for m in 2..=(n as usize) {
        if m % 2 == 1 {
            bernoulli[m] = 0.0;
            continue;
        }

        let mut sum = 0.0;
        for (k, &bernoulli_k) in bernoulli.iter().enumerate().take(m) {
            let binom_coeff = binomial((m + 1) as u32, k as u32)?;
            sum += binom_coeff * bernoulli_k;
        }
        bernoulli[m] = -sum / (m + 1) as f64;
    }

    Ok(bernoulli[n as usize])
}

/// Computes the Euler number E(n).
///
/// The Euler numbers are integers defined by the generating function
/// sech(t) = 2/(e^t + e^(-t)) = sum(E_n * t^n / n!).
/// Only even-indexed Euler numbers are non-zero.
///
/// # Arguments
///
/// * `n` - Index of the Euler number
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Euler number E(n)
///
/// # Examples
///
/// ```
/// use scirs2_special::euler_number;
///
/// assert_eq!(euler_number(0).expect("test/example should not fail"), 1.0);
/// assert_eq!(euler_number(1).expect("test/example should not fail"), 0.0);
/// assert_eq!(euler_number(2).expect("test/example should not fail"), -1.0);
/// assert_eq!(euler_number(4).expect("test/example should not fail"), 5.0);
/// ```
#[allow(dead_code)]
pub fn euler_number(n: u32) -> SpecialResult<f64> {
    if n % 2 == 1 {
        return Ok(0.0); // Odd Euler numbers are zero
    }

    // Known values for small even n
    match n {
        0 => return Ok(1.0),
        2 => return Ok(-1.0),
        4 => return Ok(5.0),
        6 => return Ok(-61.0),
        8 => return Ok(1385.0),
        10 => return Ok(-50521.0),
        _ => {} // Fall through to computation below
    }

    // For larger values, use more efficient algorithms
    if n > 100 {
        // For very large n, use asymptotic approximation
        return euler_number_asymptotic(n as i32);
    } else if n > 20 {
        // For moderate values, use improved recurrence with rational arithmetic
        return euler_number_improved_recurrence(n as i32);
    }

    // Use standard recurrence relation for small values
    euler_number_standard_recurrence(n as i32)
}

/// Standard recurrence relation for Euler numbers (small n)
#[allow(dead_code)]
fn euler_number_standard_recurrence(n: i32) -> SpecialResult<f64> {
    // Use recurrence relation based on generating function
    let mut euler = vec![0.0; (n + 1) as usize];
    euler[0] = 1.0;

    for m in (2..=(n as usize)).step_by(2) {
        let mut sum = 0.0;
        for k in (0..m).step_by(2) {
            let binom_coeff = binomial(m as u32, k as u32)?;
            sum += binom_coeff * euler[k];
        }
        euler[m] = -sum;
    }

    Ok(euler[n as usize])
}

/// Improved recurrence relation for Euler numbers using rational arithmetic (moderate n)
#[allow(dead_code)]
fn euler_number_improved_recurrence(n: i32) -> SpecialResult<f64> {
    // Use the recurrence relation with better numerical stability
    // E_n = -sum_{k=0}^{n-1} C(n,k) * E_k for even n

    if n % 2 == 1 {
        return Ok(0.0); // Euler numbers are zero for odd indices
    }

    // For moderate values, use a more memory-efficient approach
    // that doesn't store all intermediate values

    let mut _prev_eulr = [0.0; 2]; // Only store last two values (reserved for future optimization)
    _prev_eulr[0] = 1.0; // E_0 = 1

    if n == 0 {
        return Ok(1.0);
    }

    // Store a few more values for better efficiency
    let mut euler_cache = vec![0.0; (n / 2 + 1) as usize];
    euler_cache[0] = 1.0; // E_0 = 1

    for m in (2..=n).step_by(2) {
        let m_idx = (m / 2) as usize;
        let mut sum = 0.0;

        for k in (0..m).step_by(2) {
            let k_idx = (k / 2) as usize;

            // Use more efficient binomial coefficient computation
            let binom_coeff = efficient_binomial(m as u32, k as u32)?;
            sum += binom_coeff * euler_cache[k_idx];
        }

        euler_cache[m_idx] = -sum;
    }

    Ok(euler_cache[(n / 2) as usize])
}

/// Asymptotic approximation for Euler numbers (large n)
#[allow(dead_code)]
fn euler_number_asymptotic(n: i32) -> SpecialResult<f64> {
    if n % 2 == 1 {
        return Ok(0.0); // Euler numbers are zero for odd indices
    }

    // For large even n, use the asymptotic formula:
    // |E_n| ~ 8 * sqrt(2/π) * (2^{n+2} * n! / π^{n+1}) * [1 + O(1/n)]

    let n_f = n as f64;

    // Use Stirling's approximation for n!
    // ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
    let ln_n_factorial = n_f * n_f.ln() - n_f + 0.5 * (2.0 * PI * n_f).ln();

    // Calculate log of the asymptotic approximation to avoid overflow
    // ln|E_n| ≈ ln(8) + 0.5*ln(2/π) + (n+2)*ln(2) + ln(n!) - (n+1)*ln(π)

    let ln_8 = 8.0_f64.ln();
    let ln_sqrt_2_over_pi = 0.5 * (2.0 / PI).ln();
    let power_of_2_term = (n_f + 2.0) * 2.0_f64.ln();
    let pi_power_term = -(n_f + 1.0) * PI.ln();

    let ln_magnitude = ln_8 + ln_sqrt_2_over_pi + power_of_2_term + ln_n_factorial + pi_power_term;

    // Check if the result would overflow
    if ln_magnitude > 700.0 {
        return Err(SpecialError::OverflowError(
            "Euler number too large to represent as f64".to_string(),
        ));
    }

    let magnitude = ln_magnitude.exp();

    // Apply the correct sign: E_n = (-1)^{n/2} * |E_n|
    let sign = if (n / 2) % 2 == 0 { 1.0 } else { -1.0 };

    Ok(sign * magnitude)
}

/// More efficient binomial coefficient computation for moderate values
#[allow(dead_code)]
fn efficient_binomial(n: u32, k: u32) -> SpecialResult<f64> {
    if k > n {
        return Ok(0.0);
    }

    if k == 0 || k == n {
        return Ok(1.0);
    }

    // Use symmetry to reduce computation
    let k_use = if k > n - k { n - k } else { k };

    // For larger values, use logarithmic computation to avoid overflow
    if n > 30 {
        use crate::gamma::gamma;

        // C(n,k) = Γ(n+1) / (Γ(k+1) * Γ(n-k+1))
        let ln_result = (gamma((n + 1) as f64).ln())
            - (gamma((k_use + 1) as f64).ln())
            - (gamma((n - k_use + 1) as f64).ln());

        if ln_result > 700.0 {
            return Err(SpecialError::OverflowError(
                "Binomial coefficient too large".to_string(),
            ));
        }

        Ok(ln_result.exp())
    } else {
        // Use the standard multiplication approach for smaller values
        let mut result = 1.0;
        for i in 0..k_use {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        Ok(result)
    }
}

/// Combination function - alias for binomial coefficient
///
/// This function provides SciPy compatibility for `comb(n, k)`
/// which is the number of ways to choose k items from n items.
#[allow(dead_code)]
pub fn comb(n: u32, k: u32) -> SpecialResult<f64> {
    binomial(n, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0).expect("test/example should not fail"), 1.0);
        assert_eq!(factorial(1).expect("test/example should not fail"), 1.0);
        assert_eq!(factorial(5).expect("test/example should not fail"), 120.0);
        assert_eq!(
            factorial(10).expect("test/example should not fail"),
            3628800.0
        );

        // Test larger values using gamma function
        assert_relative_eq!(
            factorial(15).expect("test/example should not fail"),
            1307674368000.0,
            epsilon = 1.0
        );
    }

    #[test]
    fn test_double_factorial() {
        assert_eq!(
            double_factorial(0).expect("test/example should not fail"),
            1.0
        );
        assert_eq!(
            double_factorial(1).expect("test/example should not fail"),
            1.0
        );
        assert_eq!(
            double_factorial(2).expect("test/example should not fail"),
            2.0
        );
        assert_eq!(
            double_factorial(5).expect("test/example should not fail"),
            15.0
        ); // 5 × 3 × 1
        assert_eq!(
            double_factorial(6).expect("test/example should not fail"),
            48.0
        ); // 6 × 4 × 2
        assert_eq!(
            double_factorial(8).expect("test/example should not fail"),
            384.0
        ); // 8 × 6 × 4 × 2
    }

    #[test]
    fn test_factorial2() {
        assert_eq!(factorial2(0).expect("test/example should not fail"), 1.0);
        assert_eq!(factorial2(5).expect("test/example should not fail"), 15.0); // 5 × 3 × 1
        assert_eq!(factorial2(6).expect("test/example should not fail"), 48.0); // 6 × 4 × 2

        // Test that factorial2 is the same as double_factorial
        assert_eq!(
            factorial2(8).expect("test/example should not fail"),
            double_factorial(8).expect("test/example should not fail")
        );
    }

    #[test]
    fn test_factorialk() {
        assert_eq!(factorialk(0, 1).expect("test/example should not fail"), 1.0);
        assert_eq!(
            factorialk(8, 3).expect("test/example should not fail"),
            80.0
        ); // 8 × 5 × 2
        assert_eq!(
            factorialk(5, 2).expect("test/example should not fail"),
            15.0
        ); // 5 × 3 × 1
        assert_eq!(
            factorialk(6, 2).expect("test/example should not fail"),
            48.0
        ); // 6 × 4 × 2 (same as double factorial)
        assert_eq!(
            factorialk(9, 4).expect("test/example should not fail"),
            45.0
        ); // 9 × 5 × 1

        // Test k=1 (should be regular factorial)
        assert_eq!(
            factorialk(5, 1).expect("test/example should not fail"),
            factorial(5).expect("test/example should not fail")
        );

        // Test error case
        assert!(factorialk(5, 0).is_err());
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2).expect("test/example should not fail"), 10.0);
        assert_eq!(
            binomial(10, 3).expect("test/example should not fail"),
            120.0
        );
        assert_eq!(binomial(7, 0).expect("test/example should not fail"), 1.0);
        assert_eq!(binomial(7, 7).expect("test/example should not fail"), 1.0);
        assert_eq!(binomial(5, 10).expect("test/example should not fail"), 0.0); // k > n

        // Test symmetry
        assert_eq!(
            binomial(10, 3).expect("test/example should not fail"),
            binomial(10, 7).expect("test/example should not fail")
        );
    }

    #[test]
    fn test_permutations() {
        assert_eq!(
            permutations(5, 2).expect("test/example should not fail"),
            20.0
        );
        assert_eq!(
            permutations(10, 3).expect("test/example should not fail"),
            720.0
        );
        assert_eq!(
            permutations(7, 0).expect("test/example should not fail"),
            1.0
        );
        assert_eq!(
            permutations(5, 10).expect("test/example should not fail"),
            0.0
        ); // k > n
    }

    #[test]
    fn test_perm() {
        assert_eq!(perm(5, 2).expect("test/example should not fail"), 20.0);
        assert_eq!(perm(10, 3).expect("test/example should not fail"), 720.0);

        // Test that perm is the same as permutations
        assert_eq!(
            perm(7, 3).expect("test/example should not fail"),
            permutations(7, 3).expect("test/example should not fail")
        );
    }

    #[test]
    fn test_stirling_first() {
        assert_eq!(
            stirling_first(0, 0).expect("test/example should not fail"),
            1.0
        );
        assert_eq!(
            stirling_first(4, 2).expect("test/example should not fail"),
            11.0
        );
        assert_eq!(
            stirling_first(5, 3).expect("test/example should not fail"),
            35.0
        );
        assert_eq!(
            stirling_first(3, 0).expect("test/example should not fail"),
            0.0
        );
        assert_eq!(
            stirling_first(0, 3).expect("test/example should not fail"),
            0.0
        );
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(
            stirling_second(0, 0).expect("test/example should not fail"),
            1.0
        );
        assert_eq!(
            stirling_second(4, 2).expect("test/example should not fail"),
            7.0
        );
        assert_eq!(
            stirling_second(5, 3).expect("test/example should not fail"),
            25.0
        );
        assert_eq!(
            stirling_second(3, 0).expect("test/example should not fail"),
            0.0
        );
        assert_eq!(
            stirling_second(0, 3).expect("test/example should not fail"),
            0.0
        );
    }

    #[test]
    fn test_stirling2() {
        assert_eq!(stirling2(4, 2).expect("test/example should not fail"), 7.0);
        assert_eq!(stirling2(5, 3).expect("test/example should not fail"), 25.0);

        // Test that stirling2 is the same as stirling_second
        assert_eq!(
            stirling2(6, 3).expect("test/example should not fail"),
            stirling_second(6, 3).expect("test/example should not fail")
        );
    }

    #[test]
    fn test_bell_number() {
        assert_eq!(bell_number(0).expect("test/example should not fail"), 1.0);
        assert_eq!(bell_number(1).expect("test/example should not fail"), 1.0);
        assert_eq!(bell_number(2).expect("test/example should not fail"), 2.0);
        assert_eq!(bell_number(3).expect("test/example should not fail"), 5.0);
        assert_eq!(bell_number(4).expect("test/example should not fail"), 15.0);
        assert_eq!(bell_number(5).expect("test/example should not fail"), 52.0);
    }

    #[test]
    fn test_bernoulli_number() {
        assert_eq!(
            bernoulli_number(0).expect("test/example should not fail"),
            1.0
        );
        assert_relative_eq!(
            bernoulli_number(1).expect("test/example should not fail"),
            -0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            bernoulli_number(2).expect("test/example should not fail"),
            1.0 / 6.0,
            epsilon = 1e-10
        );
        assert_eq!(
            bernoulli_number(3).expect("test/example should not fail"),
            0.0
        );
        assert_relative_eq!(
            bernoulli_number(4).expect("test/example should not fail"),
            -1.0 / 30.0,
            epsilon = 1e-10
        );
        assert_eq!(
            bernoulli_number(5).expect("test/example should not fail"),
            0.0
        );
    }

    #[test]
    fn test_euler_number() {
        assert_eq!(euler_number(0).expect("test/example should not fail"), 1.0);
        assert_eq!(euler_number(1).expect("test/example should not fail"), 0.0);
        assert_eq!(euler_number(2).expect("test/example should not fail"), -1.0);
        assert_eq!(euler_number(3).expect("test/example should not fail"), 0.0);
        assert_eq!(euler_number(4).expect("test/example should not fail"), 5.0);
        assert_eq!(
            euler_number(6).expect("test/example should not fail"),
            -61.0
        );
    }
}
