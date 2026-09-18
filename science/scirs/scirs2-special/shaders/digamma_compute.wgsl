// GPU compute shader for digamma (psi) function calculation
// Uses series expansions and asymptotic approximations

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Constants for digamma computation
const EULER_MASCHERONI: f32 = 0.5772156649015329;

// Bernoulli numbers B_{2k}/(2k) for asymptotic expansion
const BERNOULLI_COEFFS: array<f32, 6> = array<f32, 6>(
    0.08333333333333333,    // B_2/2 = 1/12
    -0.008333333333333333,  // B_4/4 = -1/120
    0.003968253968253968,   // B_6/6 = 1/252
    -0.004166666666666667,  // B_8/8 = -1/240
    0.007575757575757576,   // B_10/10 = 1/132
    -0.021092796092796094   // B_12/12 = -691/32760
);

// Digamma function for x > 6 using asymptotic expansion
fn digamma_asymptotic(x: f32) -> f32 {
    let log_x = log(x);
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    
    var sum = log_x - 0.5 * inv_x;
    var inv_x_power = inv_x2;
    
    for (var i = 0u; i < 6u; i++) {
        sum -= BERNOULLI_COEFFS[i] * inv_x_power;
        inv_x_power *= inv_x2;
    }
    
    return sum;
}

// Digamma function using recurrence relation
fn digamma_recurrence(x: f32) -> f32 {
    // For x < 1, use reflection formula
    if (x < 1.0) {
        let pi = 3.141592653589793;
        return digamma_recurrence(1.0 - x) - pi / tan(pi * x);
    }
    
    // For 1 <= x <= 6, use recurrence to increase x
    var z = x;
    var sum = 0.0;
    
    while (z < 6.0) {
        sum -= 1.0 / z;
        z += 1.0;
    }
    
    return sum + digamma_asymptotic(z);
}

// Main digamma function
fn digamma(x: f32) -> f32 {
    // Handle special cases
    if (x <= 0.0 && x == floor(x)) {
        // Poles at negative integers and zero
        return sign(sin(3.141592653589793 * x)) * 1.0e10; // Large value as proxy for infinity
    }
    
    if (x > 6.0) {
        return digamma_asymptotic(x);
    } else {
        return digamma_recurrence(x);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    output_data[index] = digamma(x);
}