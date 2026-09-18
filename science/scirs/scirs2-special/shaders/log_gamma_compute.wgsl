// GPU compute shader for log gamma function calculation
// Uses Lanczos approximation for efficiency

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Constants for log gamma computation
const LOG_2PI: f32 = 1.8378770664093453;
const HALF_LOG_2PI: f32 = 0.9189385332046727;

// Lanczos coefficients for log gamma
const LANCZOS_G: f32 = 7.0;
const LANCZOS_COEFFS: array<f32, 9> = array<f32, 9>(
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
);

// Log gamma using Lanczos approximation
fn log_gamma_lanczos(x: f32) -> f32 {
    if (x < 0.5) {
        // Use reflection formula: log(Γ(x)) = log(π) - log(sin(πx)) - log(Γ(1-x))
        let pi = 3.141592653589793;
        let log_pi = 1.1447298858494002;
        return log_pi - log(abs(sin(pi * x))) - log_gamma_lanczos(1.0 - x);
    }
    
    let z = x - 1.0;
    var acc = LANCZOS_COEFFS[0];
    
    for (var i = 1u; i < 9u; i++) {
        acc += LANCZOS_COEFFS[i] / (z + f32(i));
    }
    
    let t = z + LANCZOS_G + 0.5;
    
    return HALF_LOG_2PI + (z + 0.5) * log(t) - t + log(acc);
}

// Optimized log gamma for special cases
fn log_gamma_optimized(x: f32) -> f32 {
    // Handle special cases
    if (x <= 0.0) {
        if (x == floor(x)) {
            // log(Γ(x)) is undefined at non-positive integers
            return 1.0e10; // Large value as proxy for infinity
        }
    }
    
    // For very small positive x, use series expansion
    if (x > 0.0 && x < 1e-5) {
        let euler_gamma = 0.5772156649015329;
        return -log(x) - euler_gamma * x;
    }
    
    // For x = 1 and x = 2, return exact values
    if (abs(x - 1.0) < 1e-10) {
        return 0.0; // log(Γ(1)) = log(0!) = 0
    }
    if (abs(x - 2.0) < 1e-10) {
        return 0.0; // log(Γ(2)) = log(1!) = 0
    }
    
    // For large x, use Stirling's approximation
    if (x > 100.0) {
        return (x - 0.5) * log(x) - x + HALF_LOG_2PI;
    }
    
    // Use Lanczos approximation for general case
    return log_gamma_lanczos(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    output_data[index] = log_gamma_optimized(x);
}