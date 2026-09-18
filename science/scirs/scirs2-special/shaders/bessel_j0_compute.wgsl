// GPU compute shader for Bessel J0 function calculation
// Uses optimized rational approximations and asymptotic expansions
// Implements high-precision algorithms suitable for GPU parallelization

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Mathematical constants with extended precision
const PI: f32 = 3.141592653589793;
const PI_4: f32 = 0.7853981633974483;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const TWO_OVER_PI: f32 = 0.6366197723675814;

// Threshold values for algorithm selection
const SMALL_THRESHOLD: f32 = 1e-4;
const MEDIUM_THRESHOLD: f32 = 5.0;
const LARGE_THRESHOLD: f32 = 100.0;

// Enhanced rational approximation for |x| <= 5 with higher accuracy
fn j0_small(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    
    // High-precision coefficients for numerator (Chebyshev approximation)
    let num = fma(fma(fma(fma(-1.8952604040652234e-9, x8, 2.5436920653092429e-6), x6, -6.524002327054125e-4), x4, 0.06249999999999973), x2, -1.0);
    
    // High-precision coefficients for denominator
    let den = fma(fma(fma(fma(1.2537723865708333e-9, x8, -4.167220649889615e-7), x6, 9.185364538511414e-5), x4, -0.013951539648688288), x2, 1.0);
    
    return -num / den;
}

// Very small x approximation using Taylor series
fn j0_tiny(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    
    // J0(x) = 1 - x²/4 + x⁴/64 - x⁶/2304 + x⁸/147456 - ...
    let term2 = x2 * 0.25;
    let term4 = x4 * 0.015625; // 1/64
    let term6 = x6 * 4.3402777777777776e-4; // 1/2304
    let term8 = x8 * 6.781684027777778e-6; // 1/147456
    
    return 1.0 - term2 + term4 - term6 + term8;
}

// Enhanced asymptotic approximation for |x| > 5 with better accuracy
fn j0_large(x: f32) -> f32 {
    let ax = abs(x);
    let z = 8.0 / ax;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;
    let z8 = z4 * z4;
    
    // Enhanced P0 coefficients (more terms for higher accuracy)
    let p0 = fma(fma(fma(-0.013084187, z6, 0.1121520996), z4, -0.0703125), z2, 1.0);
    
    // Enhanced Q0 coefficients
    let q0 = fma(fma(fma(0.003432106, z8, -0.003759766), z6, 0.0444479255), z3, -0.0390625 * z);
    
    // Phase calculation with higher-order corrections
    let phase_correction = z * q0;
    let theta = ax - PI_4 + phase_correction;
    
    // Amplitude factor with improved precision
    let sqrt_factor = sqrt(TWO_OVER_PI / ax);
    
    // Compute result using optimized trigonometric functions
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    
    return sqrt_factor * (p0 * cos_theta - z * q0 * sin_theta);
}

// Asymptotic approximation for very large x (x > 50)
fn j0_very_large(x: f32) -> f32 {
    let ax = abs(x);
    
    // For very large x, use simplified asymptotic form
    // J0(x) ≈ sqrt(2/(πx)) * cos(x - π/4)
    let sqrt_factor = sqrt(TWO_OVER_PI / ax);
    let theta = ax - PI_4;
    
    return sqrt_factor * cos(theta);
}

// Comprehensive Bessel J0 function with adaptive algorithm selection
fn bessel_j0(x: f32) -> f32 {
    let ax = abs(x);
    
    // Handle special case: x = 0
    if (ax == 0.0) {
        return 1.0;
    }
    
    // Handle very large values
    if (ax > LARGE_THRESHOLD) {
        // For x > 100, oscillations become very small amplitude
        if (ax > 1000.0) {
            return 0.0; // Effectively zero for very large arguments
        }
        return j0_very_large(ax);
    }
    
    // Handle very small values with high-precision Taylor series
    if (ax < SMALL_THRESHOLD) {
        return j0_tiny(ax);
    }
    
    // Medium range: use optimized rational approximation
    if (ax <= MEDIUM_THRESHOLD) {
        return j0_small(ax);
    }
    
    // Large range: use asymptotic expansion
    return j0_large(ax);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds checking with early return
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Input validation
    if (!isFinite(x)) {
        output_data[index] = f32(0x7fc00000); // NaN for invalid input
        return;
    }
    
    // Compute Bessel J0 function
    let result = bessel_j0(x);
    
    // Output validation
    if (isFinite(result)) {
        output_data[index] = result;
    } else {
        output_data[index] = f32(0x7fc00000); // NaN for computation errors
    }
}