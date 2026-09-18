// GPU compute shader for spherical Bessel j0 function calculation
// j0(x) = sin(x)/x with proper handling of x=0

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Spherical Bessel j0 function using series expansion for small x
fn spherical_j0_series(x: f32) -> f32 {
    // For |x| < 0.5, use Taylor series
    // j0(x) = 1 - x²/6 + x⁴/120 - x⁶/5040 + ...
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    
    return 1.0 - x2 / 6.0 + x4 / 120.0 - x6 / 5040.0 + x8 / 362880.0;
}

// Spherical Bessel j0 function using direct formula for larger x
fn spherical_j0_direct(x: f32) -> f32 {
    return sin(x) / x;
}

// Main spherical Bessel j0 function
fn spherical_j0(x: f32) -> f32 {
    let ax = abs(x);
    
    // Handle x = 0
    if (ax < 1e-10) {
        return 1.0;
    }
    
    // Use series for small |x|
    if (ax < 0.5) {
        return spherical_j0_series(x);
    }
    
    // Use direct formula for larger |x|
    return spherical_j0_direct(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    output_data[index] = spherical_j0(x);
}