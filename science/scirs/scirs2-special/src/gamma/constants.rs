//! High-precision constants for gamma function computation

/// Euler-Mascheroni constant with high precision
pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// sqrt(2π) with high precision
pub const SQRT_2PI: f64 = 2.506_628_274_631_000_7;

/// log(sqrt(2π)) with high precision
pub const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// log(2π) with high precision
#[allow(dead_code)]
pub const LOG_2PI: f64 = 1.837_877_066_409_345_6;

/// Lanczos approximation coefficients for gamma function (g=7)
/// These coefficients provide high accuracy for the gamma function
/// using the Lanczos approximation method with g=7
pub const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];
