//! Weight initialization strategies for neural network parameters.
//!
//! Provides common initialization methods including Xavier/Glorot, Kaiming/He,
//! LeCun, orthogonal, and basic constant/normal/uniform initializations.
//! Uses a deterministic LCG-based RNG (no `rand` crate dependency).

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during weight initialization.
#[derive(Debug, Clone)]
pub enum InitError {
    /// Fan-in value is invalid (zero).
    InvalidFanIn(usize),
    /// Fan-out value is invalid (zero).
    InvalidFanOut(usize),
    /// Gain value is invalid (non-positive or non-finite).
    InvalidGain(f64),
    /// Standard deviation is invalid (non-positive or non-finite).
    InvalidStd(f64),
    /// Shape is too small for the requested operation.
    ShapeTooSmall { shape: Vec<usize> },
    /// Shape is empty.
    EmptyShape,
    /// Array creation failed.
    ShapeError(String),
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFanIn(v) => write!(f, "invalid fan_in: {v}"),
            Self::InvalidFanOut(v) => write!(f, "invalid fan_out: {v}"),
            Self::InvalidGain(v) => write!(f, "invalid gain: {v}"),
            Self::InvalidStd(v) => write!(f, "invalid std: {v}"),
            Self::ShapeTooSmall { shape } => write!(f, "shape too small: {shape:?}"),
            Self::EmptyShape => write!(f, "empty shape"),
            Self::ShapeError(msg) => write!(f, "shape error: {msg}"),
        }
    }
}

impl std::error::Error for InitError {}

// ---------------------------------------------------------------------------
// FanMode
// ---------------------------------------------------------------------------

/// Selects whether to use fan_in or fan_out for Kaiming initialization.
#[derive(Debug, Clone, PartialEq)]
pub enum FanMode {
    /// Use the number of input connections (fan_in).
    FanIn,
    /// Use the number of output connections (fan_out).
    FanOut,
}

// ---------------------------------------------------------------------------
// Deterministic LCG RNG
// ---------------------------------------------------------------------------

/// A deterministic linear congruential generator (LCG) for reproducible
/// weight initialization without depending on the `rand` crate.
#[derive(Debug, Clone)]
pub struct InitRng {
    state: u64,
}

impl InitRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the LCG state by one step.
    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
    }

    /// Return the next uniform value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.step();
        (self.state >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Return a sample from the standard normal distribution N(0,1)
    /// using the Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE); // avoid ln(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Return a uniform value in `[low, high)`.
    pub fn next_uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

// ---------------------------------------------------------------------------
// Helper: compute fan_in / fan_out
// ---------------------------------------------------------------------------

/// Compute `(fan_in, fan_out)` from a weight tensor shape.
///
/// - 2-D `[out_features, in_features]`:  fan_in = in_features, fan_out = out_features
/// - N-D (N >= 3) convolution `[out_channels, in_channels, k1, k2, ...]`:
///   fan_in  = in_channels  * product(k_dims)
///   fan_out = out_channels * product(k_dims)
pub fn compute_fans(shape: &[usize]) -> Result<(usize, usize), InitError> {
    match shape.len() {
        0 => Err(InitError::EmptyShape),
        1 => Err(InitError::ShapeTooSmall {
            shape: shape.to_vec(),
        }),
        2 => {
            let fan_out = shape[0];
            let fan_in = shape[1];
            if fan_in == 0 {
                return Err(InitError::InvalidFanIn(0));
            }
            if fan_out == 0 {
                return Err(InitError::InvalidFanOut(0));
            }
            Ok((fan_in, fan_out))
        }
        _ => {
            let receptive_field: usize = shape[2..].iter().product();
            let fan_in = shape[1] * receptive_field;
            let fan_out = shape[0] * receptive_field;
            if fan_in == 0 {
                return Err(InitError::InvalidFanIn(0));
            }
            if fan_out == 0 {
                return Err(InitError::InvalidFanOut(0));
            }
            Ok((fan_in, fan_out))
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build ArrayD from a Vec
// ---------------------------------------------------------------------------

fn make_array(shape: &[usize], data: Vec<f64>) -> Result<ArrayD<f64>, InitError> {
    ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|e| InitError::ShapeError(e.to_string()))
}

fn total_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ---------------------------------------------------------------------------
// Gain helper
// ---------------------------------------------------------------------------

/// Return the recommended gain for a given activation function name.
///
/// | Activation    | Gain                              |
/// |---------------|-----------------------------------|
/// | `"linear"`    | 1.0                               |
/// | `"sigmoid"`   | 1.0                               |
/// | `"tanh"`      | 5.0 / 3.0                         |
/// | `"relu"`      | sqrt(2.0)                         |
/// | `"leaky_relu"`| sqrt(2.0 / (1 + 0.01^2))         |
/// | `"selu"`      | 0.75                              |
/// | other         | 1.0                               |
pub fn gain_for_activation(activation: &str) -> f64 {
    match activation {
        "linear" | "sigmoid" => 1.0,
        "tanh" => 5.0 / 3.0,
        "relu" => 2.0_f64.sqrt(),
        "leaky_relu" => (2.0 / (1.0 + 0.01_f64.powi(2))).sqrt(),
        "selu" => 3.0 / 4.0,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Xavier / Glorot
// ---------------------------------------------------------------------------

/// Xavier (Glorot) **uniform** initialization.
///
/// Values drawn from U(-limit, limit) where `limit = gain * sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(
    shape: &[usize],
    gain: f64,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    validate_gain(gain)?;
    let (fan_in, fan_out) = compute_fans(shape)?;
    let limit = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| rng.next_uniform(-limit, limit)).collect();
    make_array(shape, data)
}

/// Xavier (Glorot) **normal** initialization.
///
/// Values drawn from N(0, std) where `std = gain * sqrt(2 / (fan_in + fan_out))`.
pub fn xavier_normal(
    shape: &[usize],
    gain: f64,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    validate_gain(gain)?;
    let (fan_in, fan_out) = compute_fans(shape)?;
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| std * rng.next_normal()).collect();
    make_array(shape, data)
}

// ---------------------------------------------------------------------------
// Kaiming / He
// ---------------------------------------------------------------------------

/// Kaiming (He) **uniform** initialization.
///
/// Values drawn from U(-bound, bound) where `bound = gain * sqrt(3 / fan)`.
pub fn kaiming_uniform(
    shape: &[usize],
    gain: f64,
    mode: FanMode,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    validate_gain(gain)?;
    let (fan_in, fan_out) = compute_fans(shape)?;
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let bound = gain * (3.0 / fan as f64).sqrt();
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| rng.next_uniform(-bound, bound)).collect();
    make_array(shape, data)
}

/// Kaiming (He) **normal** initialization.
///
/// Values drawn from N(0, std) where `std = gain / sqrt(fan)`.
pub fn kaiming_normal(
    shape: &[usize],
    gain: f64,
    mode: FanMode,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    validate_gain(gain)?;
    let (fan_in, fan_out) = compute_fans(shape)?;
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let std = gain / (fan as f64).sqrt();
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| std * rng.next_normal()).collect();
    make_array(shape, data)
}

// ---------------------------------------------------------------------------
// LeCun
// ---------------------------------------------------------------------------

/// LeCun **normal** initialization: N(0, 1/sqrt(fan_in)).
pub fn lecun_normal(shape: &[usize], rng: &mut InitRng) -> Result<ArrayD<f64>, InitError> {
    let (fan_in, _) = compute_fans(shape)?;
    let std = 1.0 / (fan_in as f64).sqrt();
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| std * rng.next_normal()).collect();
    make_array(shape, data)
}

/// LeCun **uniform** initialization: U(-limit, limit) where `limit = sqrt(3/fan_in)`.
pub fn lecun_uniform(shape: &[usize], rng: &mut InitRng) -> Result<ArrayD<f64>, InitError> {
    let (fan_in, _) = compute_fans(shape)?;
    let limit = (3.0 / fan_in as f64).sqrt();
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| rng.next_uniform(-limit, limit)).collect();
    make_array(shape, data)
}

// ---------------------------------------------------------------------------
// Constant / Zeros / Ones
// ---------------------------------------------------------------------------

/// Initialize all elements to a constant value.
pub fn constant_init(shape: &[usize], value: f64) -> ArrayD<f64> {
    ArrayD::from_elem(IxDyn(shape), value)
}

/// Initialize all elements to zero.
pub fn zeros_init(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::zeros(IxDyn(shape))
}

/// Initialize all elements to one.
pub fn ones_init(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::ones(IxDyn(shape))
}

// ---------------------------------------------------------------------------
// Normal / Uniform
// ---------------------------------------------------------------------------

/// Normal initialization with specified mean and standard deviation.
pub fn normal_init(
    shape: &[usize],
    mean: f64,
    std: f64,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    if std <= 0.0 || !std.is_finite() {
        return Err(InitError::InvalidStd(std));
    }
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| mean + std * rng.next_normal()).collect();
    make_array(shape, data)
}

/// Uniform initialization with specified bounds `[low, high)`.
pub fn uniform_init(
    shape: &[usize],
    low: f64,
    high: f64,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    if low >= high {
        return Err(InitError::InvalidStd(high - low)); // reuse for "bad range"
    }
    let n = total_elements(shape);
    let data: Vec<f64> = (0..n).map(|_| rng.next_uniform(low, high)).collect();
    make_array(shape, data)
}

// ---------------------------------------------------------------------------
// Orthogonal
// ---------------------------------------------------------------------------

/// Orthogonal initialization via QR-like Gram-Schmidt on a random matrix.
///
/// Generates a random matrix, then orthogonalises it. For non-square shapes
/// the result is reshaped to the requested dimensions. The `gain` parameter
/// scales the resulting orthogonal matrix.
pub fn orthogonal_init(
    shape: &[usize],
    gain: f64,
    rng: &mut InitRng,
) -> Result<ArrayD<f64>, InitError> {
    validate_gain(gain)?;
    if shape.len() < 2 {
        return Err(InitError::ShapeTooSmall {
            shape: shape.to_vec(),
        });
    }

    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    if rows == 0 || cols == 0 {
        return Err(InitError::ShapeTooSmall {
            shape: shape.to_vec(),
        });
    }

    // Generate a random matrix (rows x cols)
    let n = rows * cols;
    let mut flat: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();

    // Determine whether we QR on the matrix or its transpose
    let (work_rows, work_cols, transposed) = if rows >= cols {
        (rows, cols, false)
    } else {
        (cols, rows, true)
    };

    // Build column-major representation for Gram-Schmidt
    // We work on a (work_rows x work_cols) matrix stored as columns.
    let mut columns: Vec<Vec<f64>> = if !transposed {
        // columns of the original matrix
        (0..work_cols)
            .map(|c| (0..work_rows).map(|r| flat[r * cols + c]).collect())
            .collect()
    } else {
        // columns of the transpose: rows of the original
        (0..work_cols)
            .map(|c| (0..work_rows).map(|r| flat[c * cols + r]).collect())
            .collect()
    };

    // Modified Gram-Schmidt orthogonalisation
    for i in 0..work_cols {
        // Normalise column i
        let norm = dot_vec(&columns[i], &columns[i]).sqrt();
        if norm < 1e-15 {
            // Degenerate column – fill with a canonical basis vector
            for v in columns[i].iter_mut() {
                *v = 0.0;
            }
            if i < work_rows {
                columns[i][i] = 1.0;
            }
        } else {
            for v in columns[i].iter_mut() {
                *v /= norm;
            }
        }

        // Project out column i from subsequent columns
        let qi = columns[i].clone();
        for col in columns.iter_mut().skip(i + 1) {
            let proj = dot_vec(&qi, col);
            for (v, q) in col.iter_mut().zip(qi.iter()) {
                *v -= proj * q;
            }
        }
    }

    // Reassemble into flat (rows x cols) row-major
    flat.clear();
    if !transposed {
        for r in 0..rows {
            for col in columns.iter().take(cols) {
                flat.push(gain * col[r]);
            }
        }
    } else {
        for col_vec in columns.iter().take(rows) {
            for &val in col_vec.iter().take(cols) {
                flat.push(gain * val);
            }
        }
    }

    make_array(shape, flat)
}

/// Dot product of two equal-length vectors.
fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_gain(gain: f64) -> Result<(), InitError> {
    if gain <= 0.0 || !gain.is_finite() {
        return Err(InitError::InvalidGain(gain));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// InitStats
// ---------------------------------------------------------------------------

/// Statistics about an initialized weight tensor.
#[derive(Debug, Clone)]
pub struct InitStats {
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Total number of elements.
    pub num_elements: usize,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Computed fan_in.
    pub fan_in: usize,
    /// Computed fan_out.
    pub fan_out: usize,
}

impl InitStats {
    /// Compute statistics for the given tensor and shape.
    pub fn compute(tensor: &ArrayD<f64>, shape: &[usize]) -> Self {
        let n = tensor.len();
        let (fan_in, fan_out) = compute_fans(shape).unwrap_or((0, 0));

        let mut sum = 0.0_f64;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &v in tensor.iter() {
            sum += v;
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        let mean = if n > 0 { sum / n as f64 } else { 0.0 };

        let variance = if n > 1 {
            let mut sq_sum = 0.0_f64;
            for &v in tensor.iter() {
                sq_sum += (v - mean).powi(2);
            }
            sq_sum / n as f64
        } else {
            0.0
        };

        Self {
            shape: shape.to_vec(),
            num_elements: n,
            mean,
            std: variance.sqrt(),
            min: min_val,
            max: max_val,
            fan_in,
            fan_out,
        }
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "InitStats {{ shape: {:?}, n: {}, mean: {:.6}, std: {:.6}, \
             min: {:.6}, max: {:.6}, fan_in: {}, fan_out: {} }}",
            self.shape,
            self.num_elements,
            self.mean,
            self.std,
            self.min,
            self.max,
            self.fan_in,
            self.fan_out,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fans_2d() {
        let (fan_in, fan_out) = compute_fans(&[10, 5]).expect("compute_fans failed");
        assert_eq!(fan_in, 5);
        assert_eq!(fan_out, 10);
    }

    #[test]
    fn test_compute_fans_4d() {
        // [out_ch=16, in_ch=3, kH=3, kW=3]
        let (fan_in, fan_out) = compute_fans(&[16, 3, 3, 3]).expect("compute_fans failed");
        assert_eq!(fan_in, 3 * 3 * 3); // 27
        assert_eq!(fan_out, 16 * 3 * 3); // 144
    }

    #[test]
    fn test_xavier_uniform_range() {
        let shape = [64, 32];
        let (fan_in, fan_out) = compute_fans(&shape).expect("fans");
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let mut rng = InitRng::new(42);
        let arr = xavier_uniform(&shape, 1.0, &mut rng).expect("xavier_uniform");
        for &v in arr.iter() {
            assert!(
                v >= -limit && v <= limit,
                "value {v} outside [{}, {}]",
                -limit,
                limit
            );
        }
    }

    #[test]
    fn test_xavier_normal_mean_near_zero() {
        let shape = [256, 128];
        let mut rng = InitRng::new(123);
        let arr = xavier_normal(&shape, 1.0, &mut rng).expect("xavier_normal");
        let mean: f64 = arr.iter().sum::<f64>() / arr.len() as f64;
        assert!(mean.abs() < 0.05, "mean too far from zero: {mean}");
    }

    #[test]
    fn test_kaiming_uniform_fan_in() {
        let shape = [64, 32];
        let gain = 2.0_f64.sqrt();
        let (fan_in, _) = compute_fans(&shape).expect("fans");
        let bound = gain * (3.0 / fan_in as f64).sqrt();
        let mut rng = InitRng::new(7);
        let arr = kaiming_uniform(&shape, gain, FanMode::FanIn, &mut rng).expect("kaiming_uniform");
        for &v in arr.iter() {
            assert!(
                v >= -bound && v <= bound,
                "value {v} outside [{}, {}]",
                -bound,
                bound
            );
        }
    }

    #[test]
    fn test_kaiming_normal_std() {
        let shape = [256, 128];
        let gain = 2.0_f64.sqrt();
        let (fan_in, _) = compute_fans(&shape).expect("fans");
        let expected_std = gain / (fan_in as f64).sqrt();
        let mut rng = InitRng::new(99);
        let arr = kaiming_normal(&shape, gain, FanMode::FanIn, &mut rng).expect("kaiming_normal");
        let n = arr.len() as f64;
        let mean: f64 = arr.iter().sum::<f64>() / n;
        let var: f64 = arr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let actual_std = var.sqrt();
        let ratio = actual_std / expected_std;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "std ratio {ratio} (actual={actual_std}, expected={expected_std})"
        );
    }

    #[test]
    fn test_lecun_normal_shape() {
        let shape = [16, 8, 3, 3];
        let mut rng = InitRng::new(55);
        let arr = lecun_normal(&shape, &mut rng).expect("lecun_normal");
        assert_eq!(arr.shape(), &[16, 8, 3, 3]);
    }

    #[test]
    fn test_lecun_uniform_range() {
        let shape = [32, 16];
        let (fan_in, _) = compute_fans(&shape).expect("fans");
        let limit = (3.0 / fan_in as f64).sqrt();
        let mut rng = InitRng::new(11);
        let arr = lecun_uniform(&shape, &mut rng).expect("lecun_uniform");
        for &v in arr.iter() {
            assert!(
                v >= -limit && v <= limit,
                "value {v} outside [{}, {}]",
                -limit,
                limit
            );
        }
    }

    #[test]
    fn test_constant_init_value() {
        let arr = constant_init(&[3, 4], 3.15);
        for &v in arr.iter() {
            assert!((v - 3.15).abs() < 1e-12);
        }
    }

    #[test]
    fn test_zeros_init() {
        let arr = zeros_init(&[5, 5]);
        for &v in arr.iter() {
            assert!((v).abs() < 1e-15);
        }
    }

    #[test]
    fn test_ones_init() {
        let arr = ones_init(&[2, 3]);
        for &v in arr.iter() {
            assert!((v - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_orthogonal_init_square() {
        let shape = [8, 8];
        let mut rng = InitRng::new(77);
        let arr = orthogonal_init(&shape, 1.0, &mut rng).expect("orthogonal_init");
        // Check Q^T Q ≈ I  (columns are orthonormal)
        let n = 8;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0_f64;
                for k in 0..n {
                    // arr[[k, i]] * arr[[k, j]]
                    dot += arr[[k, i].as_ref()] * arr[[k, j].as_ref()];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "Q^T Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_normal_init_distribution() {
        let shape = [512, 256];
        let target_mean = 2.0;
        let target_std = 0.5;
        let mut rng = InitRng::new(42);
        let arr = normal_init(&shape, target_mean, target_std, &mut rng).expect("normal_init");
        let n = arr.len() as f64;
        let mean: f64 = arr.iter().sum::<f64>() / n;
        let var: f64 = arr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let actual_std = var.sqrt();
        assert!(
            (mean - target_mean).abs() < 0.05,
            "mean {mean} far from {target_mean}"
        );
        assert!(
            (actual_std - target_std).abs() < 0.05,
            "std {actual_std} far from {target_std}"
        );
    }

    #[test]
    fn test_uniform_init_bounds() {
        let shape = [100, 100];
        let mut rng = InitRng::new(13);
        let arr = uniform_init(&shape, -0.5, 0.5, &mut rng).expect("uniform_init");
        for &v in arr.iter() {
            assert!((-0.5..0.5).contains(&v), "value {v} out of bounds");
        }
    }

    #[test]
    fn test_gain_for_relu() {
        let g = gain_for_activation("relu");
        assert!((g - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_gain_for_tanh() {
        let g = gain_for_activation("tanh");
        assert!((g - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_gain_for_unknown() {
        assert!((gain_for_activation("swish") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_init_stats_compute() {
        let arr = ones_init(&[4, 5]);
        let stats = InitStats::compute(&arr, &[4, 5]);
        assert_eq!(stats.num_elements, 20);
        assert!((stats.mean - 1.0).abs() < 1e-12);
        assert!(stats.std < 1e-12);
    }

    #[test]
    fn test_init_stats_summary_nonempty() {
        let arr = zeros_init(&[3, 3]);
        let stats = InitStats::compute(&arr, &[3, 3]);
        let s = stats.summary();
        assert!(!s.is_empty());
        assert!(s.contains("InitStats"));
    }

    #[test]
    fn test_fan_mode_kaiming_changes_std() {
        // For a non-square shape, fan_in != fan_out, so distributions differ.
        let shape = [128, 32];
        let gain = 2.0_f64.sqrt();

        let mut rng1 = InitRng::new(1000);
        let arr_in =
            kaiming_normal(&shape, gain, FanMode::FanIn, &mut rng1).expect("kaiming fan_in");

        let mut rng2 = InitRng::new(1000);
        let arr_out =
            kaiming_normal(&shape, gain, FanMode::FanOut, &mut rng2).expect("kaiming fan_out");

        let std_in = {
            let n = arr_in.len() as f64;
            let m: f64 = arr_in.iter().sum::<f64>() / n;
            (arr_in.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
        };
        let std_out = {
            let n = arr_out.len() as f64;
            let m: f64 = arr_out.iter().sum::<f64>() / n;
            (arr_out.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
        };

        // fan_in=32 vs fan_out=128, so std_in should be larger than std_out
        assert!(
            (std_in - std_out).abs() > 0.01,
            "std_in={std_in} and std_out={std_out} should differ significantly"
        );
    }
}
