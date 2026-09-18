//! Pooling operations for neural network tensor processing.
//!
//! Provides max pooling, average pooling, Lp pooling, global pooling,
//! adaptive pooling, and unpooling operations over N-dimensional spatial data.

use scirs2_core::ndarray::{ArrayD, IxDyn};

/// Errors that can occur during pooling operations.
#[derive(Debug, Clone)]
pub enum PoolingError {
    /// Kernel size must be > 0.
    InvalidKernelSize { size: usize },
    /// Stride must be > 0.
    InvalidStride { stride: usize },
    /// Padding must be less than kernel_size.
    InvalidPadding { padding: usize, kernel_size: usize },
    /// Input tensor does not have enough dimensions.
    InsufficientDimensions { ndim: usize, required: usize },
    /// Input tensor is empty.
    EmptyInput,
    /// Shape mismatch between tensors.
    ShapeMismatch(String),
}

impl std::fmt::Display for PoolingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKernelSize { size } => {
                write!(f, "Invalid kernel size: {size} (must be > 0)")
            }
            Self::InvalidStride { stride } => {
                write!(f, "Invalid stride: {stride} (must be > 0)")
            }
            Self::InvalidPadding {
                padding,
                kernel_size,
            } => write!(
                f,
                "Invalid padding: {padding} (must be < kernel_size {kernel_size})"
            ),
            Self::InsufficientDimensions { ndim, required } => {
                write!(
                    f,
                    "Insufficient dimensions: got {ndim}, need at least {required}"
                )
            }
            Self::EmptyInput => write!(f, "Empty input tensor"),
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for PoolingError {}

/// Pooling configuration specifying kernel size, stride, padding, and rounding mode.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Kernel (window) size for each spatial dimension.
    pub kernel_size: Vec<usize>,
    /// Stride for each spatial dimension. If empty, defaults to kernel_size.
    pub stride: Vec<usize>,
    /// Zero-padding on each side for each spatial dimension.
    pub padding: Vec<usize>,
    /// Use ceil instead of floor for output size computation.
    pub ceil_mode: bool,
}

impl PoolConfig {
    /// Create a new config with the given kernel size, stride equal to kernel size,
    /// zero padding, and floor mode.
    pub fn new(kernel_size: Vec<usize>) -> Self {
        Self {
            stride: kernel_size.clone(),
            padding: vec![0; kernel_size.len()],
            kernel_size,
            ceil_mode: false,
        }
    }

    /// Set the stride (builder pattern).
    pub fn with_stride(mut self, stride: Vec<usize>) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding (builder pattern).
    pub fn with_padding(mut self, padding: Vec<usize>) -> Self {
        self.padding = padding;
        self
    }

    /// Set ceil mode (builder pattern).
    pub fn with_ceil_mode(mut self, ceil: bool) -> Self {
        self.ceil_mode = ceil;
        self
    }

    /// Compute the output size for one spatial dimension.
    ///
    /// Formula: `floor((input + 2*padding - kernel) / stride) + 1`
    /// (or ceil if `ceil_mode` is true).
    pub fn output_size(&self, input_size: usize, dim: usize) -> usize {
        let k = self.kernel_size.get(dim).copied().unwrap_or(1);
        let s = self.effective_stride(dim);
        let p = self.padding.get(dim).copied().unwrap_or(0);
        let numerator = input_size + 2 * p;
        if numerator < k {
            return 0;
        }
        let diff = numerator - k;
        if self.ceil_mode {
            diff.div_ceil(s) + 1
        } else {
            diff / s + 1
        }
    }

    /// Validate the config, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), PoolingError> {
        for &k in &self.kernel_size {
            if k == 0 {
                return Err(PoolingError::InvalidKernelSize { size: k });
            }
        }
        for &s in &self.stride {
            if s == 0 {
                return Err(PoolingError::InvalidStride { stride: s });
            }
        }
        for (i, &p) in self.padding.iter().enumerate() {
            let k = self.kernel_size.get(i).copied().unwrap_or(1);
            if p >= k {
                return Err(PoolingError::InvalidPadding {
                    padding: p,
                    kernel_size: k,
                });
            }
        }
        Ok(())
    }

    /// Number of spatial dimensions this config covers.
    pub fn num_spatial_dims(&self) -> usize {
        self.kernel_size.len()
    }

    /// Effective stride for a given dimension (defaults to kernel_size if stride vec is short).
    fn effective_stride(&self, dim: usize) -> usize {
        self.stride
            .get(dim)
            .copied()
            .unwrap_or_else(|| self.kernel_size.get(dim).copied().unwrap_or(1))
    }

    /// Effective padding for a given dimension.
    fn effective_padding(&self, dim: usize) -> usize {
        self.padding.get(dim).copied().unwrap_or(0)
    }
}

/// Validate that input has at least `batch + channel + spatial` dimensions.
fn validate_input(input: &ArrayD<f64>, num_spatial: usize) -> Result<(), PoolingError> {
    if input.is_empty() {
        return Err(PoolingError::EmptyInput);
    }
    let required = num_spatial + 2;
    if input.ndim() < required {
        return Err(PoolingError::InsufficientDimensions {
            ndim: input.ndim(),
            required,
        });
    }
    Ok(())
}

/// Compute the output shape given input shape and pool config.
/// Returns full shape: [batch, channels, ...spatial_out...]
fn compute_output_shape(
    input_shape: &[usize],
    config: &PoolConfig,
) -> Result<Vec<usize>, PoolingError> {
    let num_spatial = config.num_spatial_dims();
    let mut out_shape = Vec::with_capacity(input_shape.len());
    // Copy batch + channel dims
    for &d in &input_shape[..input_shape.len() - num_spatial] {
        out_shape.push(d);
    }
    // Compute spatial dims
    for i in 0..num_spatial {
        let spatial_idx = input_shape.len() - num_spatial + i;
        let out = config.output_size(input_shape[spatial_idx], i);
        out_shape.push(out);
    }
    Ok(out_shape)
}

/// Iterate over all positions in the non-spatial (batch + channel) dimensions.
/// Returns the total number of "slices" and a function to map flat index → multi-dim index.
fn num_outer_slices(shape: &[usize], num_spatial: usize) -> usize {
    shape[..shape.len() - num_spatial].iter().product()
}

/// Convert a flat outer index to multi-dimensional indices for the leading dims.
fn flat_to_outer_indices(mut flat: usize, shape: &[usize], num_spatial: usize) -> Vec<usize> {
    let outer_dims = shape.len() - num_spatial;
    let mut indices = vec![0usize; outer_dims];
    for d in (0..outer_dims).rev() {
        indices[d] = flat % shape[d];
        flat /= shape[d];
    }
    indices
}

/// Extract a spatial slice from the input given outer indices.
/// Returns a view into the spatial portion.
fn get_spatial_value(
    input: &ArrayD<f64>,
    outer_indices: &[usize],
    spatial_indices: &[usize],
    num_spatial: usize,
) -> f64 {
    let ndim = input.ndim();
    let mut idx = vec![0usize; ndim];
    for (i, &oi) in outer_indices.iter().enumerate() {
        idx[i] = oi;
    }
    let offset = ndim - num_spatial;
    for (i, &si) in spatial_indices.iter().enumerate() {
        idx[offset + i] = si;
    }
    input[IxDyn(&idx)]
}

/// Iterate over all windows for the spatial dimensions, calling the callback with
/// the output spatial indices and the collected window values (with their flat spatial positions).
fn for_each_window<F>(
    input_spatial_shape: &[usize],
    config: &PoolConfig,
    output_spatial_shape: &[usize],
    mut callback: F,
) where
    F: FnMut(&[usize], Vec<(f64, Vec<usize>)>),
{
    let num_spatial = config.num_spatial_dims();
    let mut out_pos = vec![0usize; num_spatial];

    loop {
        // Collect values in the window at out_pos
        let mut window_values: Vec<(f64, Vec<usize>)> = Vec::new();
        collect_window_values(
            input_spatial_shape,
            config,
            &out_pos,
            num_spatial,
            0,
            &mut vec![0usize; num_spatial],
            &mut window_values,
        );

        callback(&out_pos, window_values);

        // Advance out_pos
        if !advance_indices(&mut out_pos, output_spatial_shape) {
            break;
        }
    }
}

/// Recursively collect all values within a pooling window.
fn collect_window_values(
    input_spatial_shape: &[usize],
    config: &PoolConfig,
    out_pos: &[usize],
    num_spatial: usize,
    dim: usize,
    current_input_pos: &mut Vec<usize>,
    results: &mut Vec<(f64, Vec<usize>)>,
) {
    if dim == num_spatial {
        // Check bounds (accounting for padding)
        let mut valid = true;
        let mut actual_pos = Vec::with_capacity(num_spatial);
        for d in 0..num_spatial {
            let p = config.effective_padding(d);
            let pos_with_pad = current_input_pos[d];
            if pos_with_pad < p || pos_with_pad >= input_spatial_shape[d] + p {
                valid = false;
                break;
            }
            actual_pos.push(pos_with_pad - p);
        }
        if valid {
            // We push a placeholder value; the caller will look it up
            results.push((0.0, actual_pos));
        }
        return;
    }

    let stride = config.effective_stride(dim);
    let k = config.kernel_size.get(dim).copied().unwrap_or(1);
    let start = out_pos[dim] * stride;

    for ki in 0..k {
        current_input_pos[dim] = start + ki;
        collect_window_values(
            input_spatial_shape,
            config,
            out_pos,
            num_spatial,
            dim + 1,
            current_input_pos,
            results,
        );
    }
}

/// Advance a multi-dimensional index. Returns false if we've wrapped around (done).
fn advance_indices(indices: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..indices.len()).rev() {
        indices[d] += 1;
        if indices[d] < shape[d] {
            return true;
        }
        indices[d] = 0;
    }
    false
}

/// Compute the flat spatial index from multi-dim spatial indices.
fn spatial_flat_index(spatial_indices: &[usize], spatial_shape: &[usize]) -> i64 {
    let mut flat: i64 = 0;
    let mut stride: i64 = 1;
    for d in (0..spatial_indices.len()).rev() {
        flat += spatial_indices[d] as i64 * stride;
        stride *= spatial_shape[d] as i64;
    }
    flat
}

/// Max pooling over spatial dimensions.
///
/// Input shape: `[batch, channels, ...spatial_dims...]`
/// Output: max over each kernel window.
pub fn max_pool(input: &ArrayD<f64>, config: &PoolConfig) -> Result<ArrayD<f64>, PoolingError> {
    config.validate()?;
    let num_spatial = config.num_spatial_dims();
    validate_input(input, num_spatial)?;

    let input_shape = input.shape();
    let out_shape = compute_output_shape(input_shape, config)?;
    let spatial_offset = input_shape.len() - num_spatial;
    let input_spatial: Vec<usize> = input_shape[spatial_offset..].to_vec();
    let output_spatial: Vec<usize> = out_shape[spatial_offset..].to_vec();

    let mut output = ArrayD::zeros(IxDyn(&out_shape));
    let n_outer = num_outer_slices(input_shape, num_spatial);

    for outer_flat in 0..n_outer {
        let outer_idx = flat_to_outer_indices(outer_flat, input_shape, num_spatial);

        for_each_window(
            &input_spatial,
            config,
            &output_spatial,
            |out_pos, positions| {
                let mut max_val = f64::NEG_INFINITY;
                for (_, actual_pos) in &positions {
                    let val = get_spatial_value(input, &outer_idx, actual_pos, num_spatial);
                    if val > max_val {
                        max_val = val;
                    }
                }
                // If no valid positions (all padding), use 0
                if max_val == f64::NEG_INFINITY {
                    max_val = 0.0;
                }
                let mut full_idx: Vec<usize> = outer_idx.clone();
                full_idx.extend_from_slice(out_pos);
                output[IxDyn(&full_idx)] = max_val;
            },
        );
    }

    Ok(output)
}

/// Max pooling with indices: returns `(pooled_output, indices_of_max)`.
///
/// The indices are flat indices into the spatial dimensions of the input.
pub fn max_pool_with_indices(
    input: &ArrayD<f64>,
    config: &PoolConfig,
) -> Result<(ArrayD<f64>, ArrayD<i64>), PoolingError> {
    config.validate()?;
    let num_spatial = config.num_spatial_dims();
    validate_input(input, num_spatial)?;

    let input_shape = input.shape();
    let out_shape = compute_output_shape(input_shape, config)?;
    let spatial_offset = input_shape.len() - num_spatial;
    let input_spatial: Vec<usize> = input_shape[spatial_offset..].to_vec();
    let output_spatial: Vec<usize> = out_shape[spatial_offset..].to_vec();

    let mut output = ArrayD::zeros(IxDyn(&out_shape));
    let mut indices = ArrayD::zeros(IxDyn(&out_shape));
    let n_outer = num_outer_slices(input_shape, num_spatial);

    for outer_flat in 0..n_outer {
        let outer_idx = flat_to_outer_indices(outer_flat, input_shape, num_spatial);

        for_each_window(
            &input_spatial,
            config,
            &output_spatial,
            |out_pos, positions| {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx: i64 = -1;
                for (_, actual_pos) in &positions {
                    let val = get_spatial_value(input, &outer_idx, actual_pos, num_spatial);
                    if val > max_val {
                        max_val = val;
                        max_idx = spatial_flat_index(actual_pos, &input_spatial);
                    }
                }
                if max_val == f64::NEG_INFINITY {
                    max_val = 0.0;
                    max_idx = 0;
                }
                let mut full_idx: Vec<usize> = outer_idx.clone();
                full_idx.extend_from_slice(out_pos);
                output[IxDyn(&full_idx)] = max_val;
                indices[IxDyn(&full_idx)] = max_idx;
            },
        );
    }

    Ok((output, indices))
}

/// Average pooling over spatial dimensions.
///
/// Input shape: `[batch, channels, ...spatial_dims...]`
pub fn avg_pool(input: &ArrayD<f64>, config: &PoolConfig) -> Result<ArrayD<f64>, PoolingError> {
    config.validate()?;
    let num_spatial = config.num_spatial_dims();
    validate_input(input, num_spatial)?;

    let input_shape = input.shape();
    let out_shape = compute_output_shape(input_shape, config)?;
    let spatial_offset = input_shape.len() - num_spatial;
    let input_spatial: Vec<usize> = input_shape[spatial_offset..].to_vec();
    let output_spatial: Vec<usize> = out_shape[spatial_offset..].to_vec();

    let mut output = ArrayD::zeros(IxDyn(&out_shape));
    let n_outer = num_outer_slices(input_shape, num_spatial);

    for outer_flat in 0..n_outer {
        let outer_idx = flat_to_outer_indices(outer_flat, input_shape, num_spatial);

        for_each_window(
            &input_spatial,
            config,
            &output_spatial,
            |out_pos, positions| {
                let mut sum = 0.0;
                let count = positions.len();
                for (_, actual_pos) in &positions {
                    sum += get_spatial_value(input, &outer_idx, actual_pos, num_spatial);
                }
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                let mut full_idx: Vec<usize> = outer_idx.clone();
                full_idx.extend_from_slice(out_pos);
                output[IxDyn(&full_idx)] = avg;
            },
        );
    }

    Ok(output)
}

/// Lp pooling (generalized): `(sum(|x|^p) / count)^(1/p)`.
pub fn lp_pool(
    input: &ArrayD<f64>,
    config: &PoolConfig,
    p: f64,
) -> Result<ArrayD<f64>, PoolingError> {
    config.validate()?;
    let num_spatial = config.num_spatial_dims();
    validate_input(input, num_spatial)?;

    let input_shape = input.shape();
    let out_shape = compute_output_shape(input_shape, config)?;
    let spatial_offset = input_shape.len() - num_spatial;
    let input_spatial: Vec<usize> = input_shape[spatial_offset..].to_vec();
    let output_spatial: Vec<usize> = out_shape[spatial_offset..].to_vec();

    let mut output = ArrayD::zeros(IxDyn(&out_shape));
    let n_outer = num_outer_slices(input_shape, num_spatial);

    for outer_flat in 0..n_outer {
        let outer_idx = flat_to_outer_indices(outer_flat, input_shape, num_spatial);

        for_each_window(
            &input_spatial,
            config,
            &output_spatial,
            |out_pos, positions| {
                let count = positions.len();
                let mut sum_pow = 0.0;
                for (_, actual_pos) in &positions {
                    let val = get_spatial_value(input, &outer_idx, actual_pos, num_spatial);
                    sum_pow += val.abs().powf(p);
                }
                let result = if count > 0 {
                    (sum_pow / count as f64).powf(1.0 / p)
                } else {
                    0.0
                };
                let mut full_idx: Vec<usize> = outer_idx.clone();
                full_idx.extend_from_slice(out_pos);
                output[IxDyn(&full_idx)] = result;
            },
        );
    }

    Ok(output)
}

/// Global max pooling: reduce all spatial dims to a single value per (batch, channel).
///
/// Input: `[batch, channels, ...spatial...]` → Output: `[batch, channels]`
pub fn global_max_pool(input: &ArrayD<f64>) -> Result<ArrayD<f64>, PoolingError> {
    if input.is_empty() {
        return Err(PoolingError::EmptyInput);
    }
    if input.ndim() < 3 {
        return Err(PoolingError::InsufficientDimensions {
            ndim: input.ndim(),
            required: 3,
        });
    }

    let shape = input.shape();
    let batch = shape[0];
    let channels = shape[1];
    let num_spatial = input.ndim() - 2;
    let spatial_size: usize = shape[2..].iter().product();

    let mut output = ArrayD::zeros(IxDyn(&[batch, channels]));

    for b in 0..batch {
        for c in 0..channels {
            let mut max_val = f64::NEG_INFINITY;
            // Iterate over all spatial positions
            for s in 0..spatial_size {
                let spatial_idx = flat_to_spatial_indices(s, &shape[2..]);
                let mut full_idx = vec![b, c];
                full_idx.extend_from_slice(&spatial_idx);
                let val = input[IxDyn(&full_idx)];
                if val > max_val {
                    max_val = val;
                }
            }
            if max_val == f64::NEG_INFINITY {
                max_val = 0.0;
            }
            output[IxDyn(&[b, c])] = max_val;
        }
    }
    // Suppress unused warning for num_spatial
    let _ = num_spatial;

    Ok(output)
}

/// Global average pooling: reduce spatial dims to mean.
///
/// Input: `[batch, channels, ...spatial...]` → Output: `[batch, channels]`
pub fn global_avg_pool(input: &ArrayD<f64>) -> Result<ArrayD<f64>, PoolingError> {
    if input.is_empty() {
        return Err(PoolingError::EmptyInput);
    }
    if input.ndim() < 3 {
        return Err(PoolingError::InsufficientDimensions {
            ndim: input.ndim(),
            required: 3,
        });
    }

    let shape = input.shape();
    let batch = shape[0];
    let channels = shape[1];
    let spatial_size: usize = shape[2..].iter().product();

    let mut output = ArrayD::zeros(IxDyn(&[batch, channels]));

    for b in 0..batch {
        for c in 0..channels {
            let mut sum = 0.0;
            for s in 0..spatial_size {
                let spatial_idx = flat_to_spatial_indices(s, &shape[2..]);
                let mut full_idx = vec![b, c];
                full_idx.extend_from_slice(&spatial_idx);
                sum += input[IxDyn(&full_idx)];
            }
            output[IxDyn(&[b, c])] = sum / spatial_size as f64;
        }
    }

    Ok(output)
}

/// Convert a flat index to multi-dimensional spatial indices.
fn flat_to_spatial_indices(mut flat: usize, spatial_shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; spatial_shape.len()];
    for d in (0..spatial_shape.len()).rev() {
        indices[d] = flat % spatial_shape[d];
        flat /= spatial_shape[d];
    }
    indices
}

/// Adaptive average pooling: automatically compute kernel/stride to achieve target output size.
///
/// Input: `[batch, channels, ...spatial...]`, `output_size` for each spatial dim.
pub fn adaptive_avg_pool(
    input: &ArrayD<f64>,
    output_size: &[usize],
) -> Result<ArrayD<f64>, PoolingError> {
    if input.is_empty() {
        return Err(PoolingError::EmptyInput);
    }
    let num_spatial = output_size.len();
    if input.ndim() < num_spatial + 2 {
        return Err(PoolingError::InsufficientDimensions {
            ndim: input.ndim(),
            required: num_spatial + 2,
        });
    }

    let shape = input.shape();
    let spatial_offset = shape.len() - num_spatial;
    let input_spatial: Vec<usize> = shape[spatial_offset..].to_vec();

    // Build output shape
    let mut out_shape: Vec<usize> = shape[..spatial_offset].to_vec();
    out_shape.extend_from_slice(output_size);

    let mut output = ArrayD::zeros(IxDyn(&out_shape));
    let n_outer = num_outer_slices(shape, num_spatial);

    for outer_flat in 0..n_outer {
        let outer_idx = flat_to_outer_indices(outer_flat, shape, num_spatial);

        // Iterate over all output spatial positions
        let mut out_pos = vec![0usize; num_spatial];
        loop {
            // For each spatial dim, compute the input range using the adaptive formula
            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(num_spatial);
            for d in 0..num_spatial {
                let in_size = input_spatial[d];
                let out_sz = output_size[d];
                let start = (out_pos[d] * in_size) / out_sz;
                let end = ((out_pos[d] + 1) * in_size) / out_sz;
                ranges.push((start, end));
            }

            // Average over the adaptive window
            let mut sum = 0.0;
            let mut count = 0usize;
            let mut win_pos = vec![0usize; num_spatial];
            // Initialize win_pos to range starts
            for d in 0..num_spatial {
                win_pos[d] = ranges[d].0;
            }
            loop {
                let val = get_spatial_value(input, &outer_idx, &win_pos, num_spatial);
                sum += val;
                count += 1;

                // Advance win_pos within ranges
                if !advance_within_ranges(&mut win_pos, &ranges) {
                    break;
                }
            }

            let avg = if count > 0 { sum / count as f64 } else { 0.0 };
            let mut full_idx: Vec<usize> = outer_idx.clone();
            full_idx.extend_from_slice(&out_pos);
            output[IxDyn(&full_idx)] = avg;

            if !advance_indices(&mut out_pos, output_size) {
                break;
            }
        }
    }

    Ok(output)
}

/// Advance indices within specified ranges (inclusive start, exclusive end).
fn advance_within_ranges(indices: &mut [usize], ranges: &[(usize, usize)]) -> bool {
    for d in (0..indices.len()).rev() {
        indices[d] += 1;
        if indices[d] < ranges[d].1 {
            return true;
        }
        indices[d] = ranges[d].0;
    }
    false
}

/// Unpool (inverse of max_pool): scatter pooled values back using stored indices.
///
/// Creates a zero tensor of `output_size` and places pooled values at the positions
/// indicated by `indices`.
pub fn max_unpool(
    pooled: &ArrayD<f64>,
    indices: &ArrayD<i64>,
    output_size: &[usize],
) -> Result<ArrayD<f64>, PoolingError> {
    if pooled.shape() != indices.shape() {
        return Err(PoolingError::ShapeMismatch(format!(
            "pooled shape {:?} != indices shape {:?}",
            pooled.shape(),
            indices.shape()
        )));
    }
    if pooled.is_empty() {
        return Err(PoolingError::EmptyInput);
    }

    let pooled_shape = pooled.shape();
    // output_size should be the full shape including batch+channel dims
    if output_size.len() != pooled_shape.len() {
        return Err(PoolingError::ShapeMismatch(format!(
            "output_size len {} != pooled ndim {}",
            output_size.len(),
            pooled_shape.len()
        )));
    }

    // Determine num_spatial by finding how many trailing dims differ
    // We assume at least 2 leading dims (batch, channel) match
    let num_spatial = pooled_shape.len().saturating_sub(2);
    let spatial_offset = pooled_shape.len() - num_spatial;
    let output_spatial: Vec<usize> = output_size[spatial_offset..].to_vec();

    let mut output = ArrayD::zeros(IxDyn(output_size));
    let n_outer = num_outer_slices(pooled_shape, num_spatial);

    // Total spatial size of output for flat index mapping
    let output_spatial_total: usize = output_spatial.iter().product();

    for outer_flat in 0..n_outer {
        let outer_idx = flat_to_outer_indices(outer_flat, pooled_shape, num_spatial);

        // Iterate over all pooled spatial positions
        let pooled_spatial: Vec<usize> = pooled_shape[spatial_offset..].to_vec();
        let mut pos = vec![0usize; num_spatial];
        loop {
            let mut pooled_full: Vec<usize> = outer_idx.clone();
            pooled_full.extend_from_slice(&pos);
            let val = pooled[IxDyn(&pooled_full)];
            let idx = indices[IxDyn(&pooled_full)];

            if idx >= 0 && (idx as usize) < output_spatial_total {
                let spatial_pos = flat_to_spatial_indices(idx as usize, &output_spatial);
                let mut out_full: Vec<usize> = outer_idx.clone();
                out_full.extend_from_slice(&spatial_pos);
                output[IxDyn(&out_full)] = val;
            }

            if !advance_indices(&mut pos, &pooled_spatial) {
                break;
            }
        }
    }

    Ok(output)
}

/// Statistics from a pooling operation.
#[derive(Debug, Clone)]
pub struct PoolingStats {
    /// Shape of the input tensor.
    pub input_shape: Vec<usize>,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
    /// Kernel size for each spatial dimension.
    pub kernel_size: Vec<usize>,
    /// Stride for each spatial dimension.
    pub stride: Vec<usize>,
    /// Total number of elements in one kernel window (product of kernel dims).
    pub receptive_field_size: usize,
    /// Ratio of input spatial elements to output spatial elements.
    pub compression_ratio: f64,
    /// Overlap ratio: how much windows overlap (0 = no overlap).
    pub overlap_ratio: f64,
}

impl PoolingStats {
    /// Compute pooling statistics from input shape and config.
    pub fn compute(input_shape: &[usize], config: &PoolConfig) -> Result<Self, PoolingError> {
        config.validate()?;
        let num_spatial = config.num_spatial_dims();
        if input_shape.len() < num_spatial + 2 {
            return Err(PoolingError::InsufficientDimensions {
                ndim: input_shape.len(),
                required: num_spatial + 2,
            });
        }

        let output_shape = compute_output_shape(input_shape, config)?;
        let spatial_offset = input_shape.len() - num_spatial;

        let input_spatial_size: usize = input_shape[spatial_offset..].iter().product();
        let output_spatial_size: usize = output_shape[spatial_offset..].iter().product();

        let receptive_field_size: usize = config.kernel_size.iter().product();

        let compression_ratio = if output_spatial_size > 0 {
            input_spatial_size as f64 / output_spatial_size as f64
        } else {
            f64::INFINITY
        };

        // Overlap ratio: for each dim, overlap = (kernel - stride) / kernel
        // Average across dims, clamped to [0, 1]
        let mut overlap_sum = 0.0;
        for d in 0..num_spatial {
            let k = config.kernel_size.get(d).copied().unwrap_or(1) as f64;
            let s = config.effective_stride(d) as f64;
            let overlap = ((k - s) / k).max(0.0);
            overlap_sum += overlap;
        }
        let overlap_ratio = if num_spatial > 0 {
            overlap_sum / num_spatial as f64
        } else {
            0.0
        };

        let effective_stride: Vec<usize> = (0..num_spatial)
            .map(|d| config.effective_stride(d))
            .collect();

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_shape,
            kernel_size: config.kernel_size.clone(),
            stride: effective_stride,
            receptive_field_size,
            compression_ratio,
            overlap_ratio,
        })
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "Pooling: {:?} -> {:?}, kernel={:?}, stride={:?}, \
             receptive_field={}, compression={:.2}x, overlap={:.2}",
            self.input_shape,
            self.output_shape,
            self.kernel_size,
            self.stride,
            self.receptive_field_size,
            self.compression_ratio,
            self.overlap_ratio,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    fn make_4d(data: Vec<f64>, h: usize, w: usize) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[1, 1, h, w]), data)
            .expect("test tensor creation should succeed")
    }

    #[test]
    fn test_pool_config_output_size() {
        let config = PoolConfig::new(vec![2, 2]);
        assert_eq!(config.output_size(4, 0), 2);
        assert_eq!(config.output_size(4, 1), 2);
    }

    #[test]
    fn test_pool_config_output_size_with_padding() {
        let config = PoolConfig::new(vec![2, 2]).with_padding(vec![1, 1]);
        // (4 + 2*1 - 2) / 2 + 1 = 4/2 + 1 = 3
        assert_eq!(config.output_size(4, 0), 3);
    }

    #[test]
    fn test_pool_config_validate_valid() {
        let config = PoolConfig::new(vec![2, 2]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pool_config_validate_zero_kernel() {
        let config = PoolConfig::new(vec![0, 2]);
        let err = config.validate();
        assert!(err.is_err());
        match err {
            Err(PoolingError::InvalidKernelSize { size: 0 }) => {}
            other => panic!("Expected InvalidKernelSize, got {:?}", other),
        }
    }

    #[test]
    fn test_max_pool_basic() {
        // 4x4 input with known values
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_4d(data, 4, 4);
        let config = PoolConfig::new(vec![2, 2]);
        let output = max_pool(&input, &config).expect("max_pool should succeed");

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output[IxDyn(&[0, 0, 0, 0])], 6.0);
        assert_eq!(output[IxDyn(&[0, 0, 0, 1])], 8.0);
        assert_eq!(output[IxDyn(&[0, 0, 1, 0])], 14.0);
        assert_eq!(output[IxDyn(&[0, 0, 1, 1])], 16.0);
    }

    #[test]
    fn test_max_pool_with_indices_correct() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_4d(data, 4, 4);
        let config = PoolConfig::new(vec![2, 2]);
        let (output, indices) =
            max_pool_with_indices(&input, &config).expect("max_pool_with_indices should succeed");

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        // Max of top-left 2x2 is 6.0 at position (1,1) -> flat index 5
        assert_eq!(output[IxDyn(&[0, 0, 0, 0])], 6.0);
        assert_eq!(indices[IxDyn(&[0, 0, 0, 0])], 5);
        // Max of top-right 2x2 is 8.0 at position (1,3) -> flat index 7
        assert_eq!(output[IxDyn(&[0, 0, 0, 1])], 8.0);
        assert_eq!(indices[IxDyn(&[0, 0, 0, 1])], 7);
        // Max of bottom-left 2x2 is 14.0 at position (3,1) -> flat index 13
        assert_eq!(output[IxDyn(&[0, 0, 1, 0])], 14.0);
        assert_eq!(indices[IxDyn(&[0, 0, 1, 0])], 13);
        // Max of bottom-right 2x2 is 16.0 at position (3,3) -> flat index 15
        assert_eq!(output[IxDyn(&[0, 0, 1, 1])], 16.0);
        assert_eq!(indices[IxDyn(&[0, 0, 1, 1])], 15);
    }

    #[test]
    fn test_avg_pool_basic() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_4d(data, 4, 4);
        let config = PoolConfig::new(vec![2, 2]);
        let output = avg_pool(&input, &config).expect("avg_pool should succeed");

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        // avg of [1,2,5,6] = 3.5
        assert!((output[IxDyn(&[0, 0, 0, 0])] - 3.5).abs() < 1e-10);
        // avg of [3,4,7,8] = 5.5
        assert!((output[IxDyn(&[0, 0, 0, 1])] - 5.5).abs() < 1e-10);
        // avg of [9,10,13,14] = 11.5
        assert!((output[IxDyn(&[0, 0, 1, 0])] - 11.5).abs() < 1e-10);
        // avg of [11,12,15,16] = 13.5
        assert!((output[IxDyn(&[0, 0, 1, 1])] - 13.5).abs() < 1e-10);
    }

    #[test]
    fn test_avg_pool_padding() {
        // With padding=1, kernel=2, stride=2, input=4 → output = (4+2-2)/2 + 1 = 3
        let data = vec![1.0; 16];
        let input = make_4d(data, 4, 4);
        let config = PoolConfig::new(vec![2, 2]).with_padding(vec![1, 1]);
        let output = avg_pool(&input, &config).expect("avg_pool with padding should succeed");

        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_lp_pool_p2() {
        // L2 pool: sqrt(mean of squares)
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let input = make_4d(data, 2, 2);
        let config = PoolConfig::new(vec![2, 2]);
        let output = lp_pool(&input, &config, 2.0).expect("lp_pool p=2 should succeed");

        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        // sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5)
        let expected = (7.5_f64).sqrt();
        assert!((output[IxDyn(&[0, 0, 0, 0])] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lp_pool_p1() {
        // L1 pool: (mean of |x|^1)^(1/1) = mean of |x|
        #[rustfmt::skip]
        let data = vec![
            1.0, -2.0,
            3.0, -4.0,
        ];
        let input = make_4d(data, 2, 2);
        let config = PoolConfig::new(vec![2, 2]);
        let output = lp_pool(&input, &config, 1.0).expect("lp_pool p=1 should succeed");

        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        // mean of [1, 2, 3, 4] = 2.5
        assert!((output[IxDyn(&[0, 0, 0, 0])] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_global_max_pool_shape() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4]));
        let output = global_max_pool(&input).expect("global_max_pool should succeed");
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_global_max_pool_values() {
        let mut input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4]));
        // Set a known max in each channel
        input[IxDyn(&[0, 0, 2, 3])] = 42.0;
        input[IxDyn(&[0, 1, 0, 0])] = 99.0;
        input[IxDyn(&[0, 2, 3, 3])] = -1.0; // all zeros except this, but 0 > -1
                                            // Channel 2: all zeros, so max = 0

        let output = global_max_pool(&input).expect("global_max_pool should succeed");
        assert_eq!(output[IxDyn(&[0, 0])], 42.0);
        assert_eq!(output[IxDyn(&[0, 1])], 99.0);
        assert_eq!(output[IxDyn(&[0, 2])], 0.0); // max of zeros and -1 is 0
    }

    #[test]
    fn test_global_avg_pool_shape() {
        let input = ArrayD::zeros(IxDyn(&[1, 3, 4, 4]));
        let output = global_avg_pool(&input).expect("global_avg_pool should succeed");
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_global_avg_pool_values() {
        let mut input = ArrayD::ones(IxDyn(&[1, 2, 2, 2]));
        // Channel 0: all ones → mean = 1.0
        // Channel 1: set all to 2.0
        input[IxDyn(&[0, 1, 0, 0])] = 2.0;
        input[IxDyn(&[0, 1, 0, 1])] = 2.0;
        input[IxDyn(&[0, 1, 1, 0])] = 2.0;
        input[IxDyn(&[0, 1, 1, 1])] = 2.0;

        let output = global_avg_pool(&input).expect("global_avg_pool should succeed");
        assert!((output[IxDyn(&[0, 0])] - 1.0).abs() < 1e-10);
        assert!((output[IxDyn(&[0, 1])] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_avg_pool_output_size() {
        let input = ArrayD::ones(IxDyn(&[1, 1, 4, 4]));
        let output = adaptive_avg_pool(&input, &[2, 2]).expect("adaptive_avg_pool should succeed");
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_adaptive_avg_pool_identity() {
        // Target same as input → should preserve values
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_4d(data.clone(), 4, 4);
        let output =
            adaptive_avg_pool(&input, &[4, 4]).expect("adaptive_avg_pool identity should succeed");
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        for (i, &v) in data.iter().enumerate() {
            let h = i / 4;
            let w = i % 4;
            assert!(
                (output[IxDyn(&[0, 0, h, w])] - v).abs() < 1e-10,
                "mismatch at ({}, {})",
                h,
                w
            );
        }
    }

    #[test]
    fn test_max_unpool_basic() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_4d(data, 4, 4);
        let config = PoolConfig::new(vec![2, 2]);

        let (pooled, indices) =
            max_pool_with_indices(&input, &config).expect("max_pool_with_indices should succeed");

        let unpooled =
            max_unpool(&pooled, &indices, &[1, 1, 4, 4]).expect("max_unpool should succeed");

        assert_eq!(unpooled.shape(), &[1, 1, 4, 4]);
        // Values at max positions should be restored
        assert_eq!(unpooled[IxDyn(&[0, 0, 1, 1])], 6.0); // index 5 → (1,1)
        assert_eq!(unpooled[IxDyn(&[0, 0, 1, 3])], 8.0); // index 7 → (1,3)
        assert_eq!(unpooled[IxDyn(&[0, 0, 3, 1])], 14.0); // index 13 → (3,1)
        assert_eq!(unpooled[IxDyn(&[0, 0, 3, 3])], 16.0); // index 15 → (3,3)
                                                          // Non-max positions should be zero
        assert_eq!(unpooled[IxDyn(&[0, 0, 0, 0])], 0.0);
        assert_eq!(unpooled[IxDyn(&[0, 0, 2, 2])], 0.0);
    }

    #[test]
    fn test_pooling_stats_compression() {
        let config = PoolConfig::new(vec![2, 2]);
        let stats =
            PoolingStats::compute(&[1, 1, 4, 4], &config).expect("stats compute should succeed");
        assert_eq!(stats.output_shape, vec![1, 1, 2, 2]);
        // 4*4 / 2*2 = 16/4 = 4.0
        assert!((stats.compression_ratio - 4.0).abs() < 1e-10);
        assert_eq!(stats.receptive_field_size, 4);
        // stride == kernel → no overlap
        assert!((stats.overlap_ratio - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pooling_stats_summary() {
        let config = PoolConfig::new(vec![2, 2]);
        let stats =
            PoolingStats::compute(&[1, 1, 4, 4], &config).expect("stats compute should succeed");
        let summary = stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Pooling"));
    }

    #[test]
    fn test_pooling_error_display() {
        let errors = vec![
            PoolingError::InvalidKernelSize { size: 0 },
            PoolingError::InvalidStride { stride: 0 },
            PoolingError::InvalidPadding {
                padding: 3,
                kernel_size: 2,
            },
            PoolingError::InsufficientDimensions {
                ndim: 2,
                required: 4,
            },
            PoolingError::EmptyInput,
            PoolingError::ShapeMismatch("test".to_string()),
        ];
        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "Error display should not be empty");
        }
    }
}
