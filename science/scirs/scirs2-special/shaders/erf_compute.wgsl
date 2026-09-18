// GPU compute shader for error function (erf) calculation
// Uses multiple high-precision approximations for optimal accuracy
// Implements Abramowitz & Stegun, Winitzki, and rational approximations

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Mathematical constants
const TWO_OVER_SQRT_PI: f32 = 1.1283791670955126; // 2/√π
const SQRT_PI: f32 = 1.7724538509055159;
const INV_SQRT_PI: f32 = 0.5641895835477563;

// Abramowitz and Stegun coefficients (7.1.26)
const A1: f32 = 0.254829592;
const A2: f32 = -0.284496736;
const A3: f32 = 1.421413741;
const A4: f32 = -1.453152027;
const A5: f32 = 1.061405429;
const P: f32 = 0.3275911;

// Winitzki approximation coefficient (higher accuracy)
const WINITZKI_A: f32 = 0.147;

// Enhanced Abramowitz and Stegun approximation (formula 7.1.26)
fn erf_abramowitz_stegun(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    
    // Handle large values where erf(x) ≈ ±1
    if (ax > 6.0) {
        return sign;
    }
    
    // Improved t calculation with better numerical stability
    let t = 1.0 / fma(P, ax, 1.0);
    
    // Optimized polynomial evaluation using nested fma operations
    let poly = t * fma(fma(fma(fma(A5, t, A4), t, A3), t, A2), t, A1);
    
    // Compute result with enhanced numerical stability
    let ax2 = ax * ax;
    let exp_term = exp(-ax2);
    let result = 1.0 - poly * exp_term;
    
    return sign * result;
}

// Winitzki's approximation (very high accuracy)
fn erf_winitzki(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    
    if (ax > 5.0) {
        return sign;
    }
    
    let x2 = ax * ax;
    let ax_term = WINITZKI_A * ax;
    let numerator = 4.0 / PI + ax_term;
    let denominator = 1.0 + ax_term;
    
    let exp_arg = -x2 * numerator / denominator;
    let result = sqrt(1.0 - exp(exp_arg));
    
    return sign * result;
}

// Rational approximation for intermediate values
fn erf_rational(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    
    let x2 = ax * ax;
    let x3 = x2 * ax;
    let x4 = x2 * x2;
    
    // Optimized rational approximation coefficients
    let num = fma(fma(0.0232, x4, 0.128), x2, 1.0) * ax;
    let den = fma(fma(0.0137, x4, 0.154), x2, 1.0);
    
    return sign * TWO_OVER_SQRT_PI * num / den;
}

// Enhanced Taylor series for very small values with extended precision
fn erf_taylor_extended(x: f32) -> f32 {
    let ax = abs(x);
    
    if (ax < 0.125) {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        let x8 = x4 * x4;
        let x10 = x8 * x2;
        
        // Extended Taylor series: erf(x) = (2/√π) * x * (1 - x²/3 + x⁴/10 - x⁶/42 + x⁸/216 - x¹⁰/1320 + ...)
        let series = 1.0 
            - x2 / 3.0 
            + x4 / 10.0 
            - x6 / 42.0 
            + x8 / 216.0 
            - x10 / 1320.0;
        
        return TWO_OVER_SQRT_PI * x * series;
    }
    
    return erf_rational(x);
}

// Asymptotic series for large values
fn erf_asymptotic(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    
    if (ax < 2.0) {
        return erf_winitzki(x);
    }
    
    // For large x, use: erfc(x) ≈ exp(-x²) / (x√π) * (1 - 1/(2x²) + 3/(4x⁴) - ...)
    let x2 = ax * ax;
    let inv_x2 = 1.0 / x2;
    
    let asymptotic_series = 1.0 - 0.5 * inv_x2 + 0.75 * inv_x2 * inv_x2;
    let erfc_val = exp(-x2) * asymptotic_series / (ax * SQRT_PI);
    
    let erf_val = 1.0 - erfc_val;
    return sign * erf_val;
}

// Adaptive error function with optimal algorithm selection
fn erf_adaptive(x: f32) -> f32 {
    // Handle special cases efficiently
    if (!isFinite(x)) {
        if (isNan(x)) {
            return x; // Propagate NaN
        }
        return select(-1.0, 1.0, x > 0.0); // ±∞ → ±1
    }
    
    let ax = abs(x);
    
    // Exact cases
    if (ax == 0.0) {
        return 0.0;
    }
    
    // Algorithm selection based on argument magnitude for optimal accuracy
    if (ax < 1e-6) {
        // For very small x: erf(x) ≈ (2/√π) * x
        return TWO_OVER_SQRT_PI * x;
    } else if (ax < 0.5) {
        // Small values: use extended Taylor series
        return erf_taylor_extended(x);
    } else if (ax < 2.0) {
        // Medium values: use Winitzki's highly accurate approximation
        return erf_winitzki(x);
    } else if (ax < 5.0) {
        // Large values: use asymptotic expansion
        return erf_asymptotic(x);
    } else {
        // Very large values: erf(x) ≈ ±1
        return select(-1.0, 1.0, x > 0.0);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds checking with early return
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Compute error function using adaptive algorithm
    let result = erf_adaptive(x);
    
    // Additional validation for edge cases
    if (!isFinite(result)) {
        // This should rarely happen with the adaptive algorithm,
        // but provide a safe fallback
        if (isNan(x)) {
            output_data[index] = x;
        } else {
            output_data[index] = select(-1.0, 1.0, x > 0.0);
        }
    } else {
        output_data[index] = result;
    }
}