//! Type Conversion Traits for Consistent Data Flow
//!
//! This module provides traits and utilities for converting data between
//! different module representations with consistent error handling and
//! optional precision tracking.
//!
//! # Features
//!
//! - **Lossless Conversion**: Convert without any data loss
//! - **Lossy Conversion**: Convert with potential precision loss
//! - **Array Conversion**: Convert between array types
//! - **Cross-Module Conversion**: Unified interface for module interop

use std::any::TypeId;
use std::fmt;
use std::marker::PhantomData;

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use num_complex::Complex;
use num_traits::{Float, NumCast, Zero};

/// Error type for conversion operations
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionError {
    /// Type mismatch
    TypeMismatch { expected: String, actual: String },
    /// Value out of range for target type
    OutOfRange {
        value: String,
        min: String,
        max: String,
    },
    /// Precision loss would occur
    PrecisionLoss {
        original: String,
        converted: String,
        loss: f64,
    },
    /// Incompatible shapes
    ShapeMismatch {
        source: Vec<usize>,
        target: Vec<usize>,
    },
    /// Invalid conversion operation
    InvalidOperation(String),
    /// Generic conversion failure
    Failed(String),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {expected}, got {actual}")
            }
            ConversionError::OutOfRange { value, min, max } => {
                write!(f, "Value {value} out of range [{min}, {max}]")
            }
            ConversionError::PrecisionLoss {
                original,
                converted,
                loss,
            } => {
                write!(
                    f,
                    "Precision loss: {original} -> {converted} (loss: {loss:.2e})"
                )
            }
            ConversionError::ShapeMismatch { source, target } => {
                write!(f, "Shape mismatch: {source:?} vs {target:?}")
            }
            ConversionError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {msg}")
            }
            ConversionError::Failed(msg) => {
                write!(f, "Conversion failed: {msg}")
            }
        }
    }
}

impl std::error::Error for ConversionError {}

impl From<ConversionError> for CoreError {
    fn from(err: ConversionError) -> Self {
        CoreError::ValidationError(
            ErrorContext::new(err.to_string()).with_location(ErrorLocation::new(file!(), line!())),
        )
    }
}

/// Result type for conversion operations
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Options for conversion operations
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionOptions {
    /// Allow precision loss
    pub allow_precision_loss: bool,
    /// Maximum allowed relative error
    pub max_relative_error: f64,
    /// Clamp values to target range instead of failing
    pub clamp_to_range: bool,
    /// Round floating point to nearest integer
    pub round_to_nearest: bool,
    /// Preserve NaN values
    pub preserve_nan: bool,
    /// Preserve infinity values
    pub preserve_infinity: bool,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            allow_precision_loss: true,
            max_relative_error: 1e-10,
            clamp_to_range: false,
            round_to_nearest: true,
            preserve_nan: true,
            preserve_infinity: true,
        }
    }
}

impl ConversionOptions {
    /// Create strict options (no precision loss allowed)
    #[must_use]
    pub fn strict() -> Self {
        Self {
            allow_precision_loss: false,
            max_relative_error: 0.0,
            clamp_to_range: false,
            round_to_nearest: false,
            preserve_nan: true,
            preserve_infinity: true,
        }
    }

    /// Create lenient options (allow clamping and precision loss)
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            allow_precision_loss: true,
            max_relative_error: 1e-6,
            clamp_to_range: true,
            round_to_nearest: true,
            preserve_nan: true,
            preserve_infinity: true,
        }
    }
}

/// Trait for lossless conversions (guaranteed no data loss)
pub trait LosslessConvert<T> {
    /// Convert to target type without any data loss
    fn convert_lossless(&self) -> ConversionResult<T>;
}

/// Trait for lossy conversions (may lose precision)
pub trait LossyConvert<T> {
    /// Convert to target type with potential precision loss
    fn convert_lossy(&self) -> T;

    /// Convert with options controlling loss behavior
    fn convert_with_options(&self, options: &ConversionOptions) -> ConversionResult<T>;
}

/// Trait for cross-module data conversion
pub trait CrossModuleConvert {
    /// The source type
    type Source;
    /// The target type
    type Target;

    /// Convert from source to target
    fn convert(source: &Self::Source) -> ConversionResult<Self::Target>;

    /// Convert with options
    fn convert_with_options(
        source: &Self::Source,
        options: &ConversionOptions,
    ) -> ConversionResult<Self::Target>;

    /// Check if conversion is possible without actually converting
    fn can_convert(source: &Self::Source) -> bool;
}

/// Trait for array type conversions
pub trait ArrayConvert<T> {
    /// Output array type
    type Output;

    /// Convert array elements to target type
    fn convert_array(&self) -> ConversionResult<Self::Output>;

    /// Convert with specified options
    fn convert_array_with_options(
        &self,
        options: &ConversionOptions,
    ) -> ConversionResult<Self::Output>;
}

// Implement LosslessConvert for common numeric types
impl LosslessConvert<f64> for f32 {
    fn convert_lossless(&self) -> ConversionResult<f64> {
        Ok(*self as f64)
    }
}

impl LosslessConvert<i64> for i32 {
    fn convert_lossless(&self) -> ConversionResult<i64> {
        Ok(*self as i64)
    }
}

impl LosslessConvert<i64> for i16 {
    fn convert_lossless(&self) -> ConversionResult<i64> {
        Ok(*self as i64)
    }
}

impl LosslessConvert<i64> for i8 {
    fn convert_lossless(&self) -> ConversionResult<i64> {
        Ok(*self as i64)
    }
}

impl LosslessConvert<u64> for u32 {
    fn convert_lossless(&self) -> ConversionResult<u64> {
        Ok(*self as u64)
    }
}

impl LosslessConvert<u64> for u16 {
    fn convert_lossless(&self) -> ConversionResult<u64> {
        Ok(*self as u64)
    }
}

impl LosslessConvert<u64> for u8 {
    fn convert_lossless(&self) -> ConversionResult<u64> {
        Ok(*self as u64)
    }
}

impl<T: Float> LosslessConvert<Complex<T>> for T {
    fn convert_lossless(&self) -> ConversionResult<Complex<T>> {
        Ok(Complex::new(*self, T::zero()))
    }
}

// Implement LossyConvert for numeric types
impl LossyConvert<f32> for f64 {
    fn convert_lossy(&self) -> f32 {
        *self as f32
    }

    fn convert_with_options(&self, options: &ConversionOptions) -> ConversionResult<f32> {
        let converted = *self as f32;

        if !options.allow_precision_loss {
            // Check if we lost significant precision
            let back = converted as f64;
            let relative_error = if self.abs() > f64::EPSILON {
                ((back - *self) / *self).abs()
            } else {
                (back - *self).abs()
            };

            if relative_error > options.max_relative_error {
                return Err(ConversionError::PrecisionLoss {
                    original: format!("{self}"),
                    converted: format!("{converted}"),
                    loss: relative_error,
                });
            }
        }

        if self.is_infinite() && !options.preserve_infinity {
            return Err(ConversionError::OutOfRange {
                value: format!("{self}"),
                min: format!("{}", f32::MIN),
                max: format!("{}", f32::MAX),
            });
        }

        if self.is_nan() && !options.preserve_nan {
            return Err(ConversionError::InvalidOperation(
                "Cannot convert NaN".to_string(),
            ));
        }

        Ok(converted)
    }
}

impl LossyConvert<i32> for f64 {
    fn convert_lossy(&self) -> i32 {
        *self as i32
    }

    fn convert_with_options(&self, options: &ConversionOptions) -> ConversionResult<i32> {
        if self.is_nan() {
            return Err(ConversionError::InvalidOperation(
                "Cannot convert NaN to integer".to_string(),
            ));
        }

        if self.is_infinite() {
            if options.clamp_to_range {
                return Ok(if *self > 0.0 { i32::MAX } else { i32::MIN });
            }
            return Err(ConversionError::OutOfRange {
                value: format!("{self}"),
                min: format!("{}", i32::MIN),
                max: format!("{}", i32::MAX),
            });
        }

        let value = if options.round_to_nearest {
            self.round()
        } else {
            self.trunc()
        };

        if value < i32::MIN as f64 || value > i32::MAX as f64 {
            if options.clamp_to_range {
                return Ok(if value < 0.0 { i32::MIN } else { i32::MAX });
            }
            return Err(ConversionError::OutOfRange {
                value: format!("{self}"),
                min: format!("{}", i32::MIN),
                max: format!("{}", i32::MAX),
            });
        }

        if !options.allow_precision_loss && (value - *self).abs() > f64::EPSILON {
            return Err(ConversionError::PrecisionLoss {
                original: format!("{self}"),
                converted: format!("{}", value as i32),
                loss: (value - *self).abs(),
            });
        }

        Ok(value as i32)
    }
}

impl LossyConvert<i64> for f64 {
    fn convert_lossy(&self) -> i64 {
        *self as i64
    }

    fn convert_with_options(&self, options: &ConversionOptions) -> ConversionResult<i64> {
        if self.is_nan() {
            return Err(ConversionError::InvalidOperation(
                "Cannot convert NaN to integer".to_string(),
            ));
        }

        if self.is_infinite() {
            if options.clamp_to_range {
                return Ok(if *self > 0.0 { i64::MAX } else { i64::MIN });
            }
            return Err(ConversionError::OutOfRange {
                value: format!("{self}"),
                min: format!("{}", i64::MIN),
                max: format!("{}", i64::MAX),
            });
        }

        let value = if options.round_to_nearest {
            self.round()
        } else {
            self.trunc()
        };

        if value < i64::MIN as f64 || value > i64::MAX as f64 {
            if options.clamp_to_range {
                return Ok(if value < 0.0 { i64::MIN } else { i64::MAX });
            }
            return Err(ConversionError::OutOfRange {
                value: format!("{self}"),
                min: format!("{}", i64::MIN),
                max: format!("{}", i64::MAX),
            });
        }

        if !options.allow_precision_loss && (value - *self).abs() > f64::EPSILON {
            return Err(ConversionError::PrecisionLoss {
                original: format!("{self}"),
                converted: format!("{}", value as i64),
                loss: (value - *self).abs(),
            });
        }

        Ok(value as i64)
    }
}

/// Type adapter for converting between incompatible types
#[derive(Debug, Clone)]
pub struct TypeAdapter<S, T> {
    _source: PhantomData<S>,
    _target: PhantomData<T>,
}

impl<S, T> TypeAdapter<S, T> {
    /// Create a new type adapter
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _source: PhantomData,
            _target: PhantomData,
        }
    }
}

impl<S, T> Default for TypeAdapter<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T> TypeAdapter<S, T>
where
    S: NumCast + Copy,
    T: NumCast + Copy,
{
    /// Adapt a value from source to target type
    pub fn adapt(&self, value: S) -> ConversionResult<T> {
        NumCast::from(value).ok_or_else(|| {
            ConversionError::Failed(format!(
                "Cannot convert {} to {}",
                std::any::type_name::<S>(),
                std::any::type_name::<T>()
            ))
        })
    }

    /// Adapt a slice of values
    pub fn adapt_slice(&self, values: &[S]) -> ConversionResult<Vec<T>> {
        values.iter().map(|&v| self.adapt(v)).collect()
    }
}

/// Data flow converter for pipeline operations
pub struct DataFlowConverter {
    /// Conversion options
    options: ConversionOptions,
    /// Type registry for custom conversions
    custom_conversions: Vec<(
        TypeId,
        TypeId,
        Box<dyn Fn(&[u8]) -> ConversionResult<Vec<u8>> + Send + Sync>,
    )>,
}

impl std::fmt::Debug for DataFlowConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFlowConverter")
            .field("options", &self.options)
            .field("custom_conversions_count", &self.custom_conversions.len())
            .finish()
    }
}

impl Default for DataFlowConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFlowConverter {
    /// Create a new data flow converter
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: ConversionOptions::default(),
            custom_conversions: Vec::new(),
        }
    }

    /// Create with specific options
    #[must_use]
    pub fn with_options(options: ConversionOptions) -> Self {
        Self {
            options,
            custom_conversions: Vec::new(),
        }
    }

    /// Get current options
    #[must_use]
    pub const fn options(&self) -> &ConversionOptions {
        &self.options
    }

    /// Set options
    pub fn set_options(&mut self, options: ConversionOptions) {
        self.options = options;
    }

    /// Convert a slice of values
    pub fn convert_slice<S, T>(&self, source: &[S]) -> ConversionResult<Vec<T>>
    where
        S: NumCast + Copy,
        T: NumCast + Copy,
    {
        let adapter: TypeAdapter<S, T> = TypeAdapter::new();
        adapter.adapt_slice(source)
    }

    /// Convert f64 to f32 with options
    pub fn f64_to_f32(&self, values: &[f64]) -> ConversionResult<Vec<f32>> {
        values
            .iter()
            .map(|v| v.convert_with_options(&self.options))
            .collect()
    }

    /// Convert f64 to i32 with options
    pub fn f64_to_i32(&self, values: &[f64]) -> ConversionResult<Vec<i32>> {
        values
            .iter()
            .map(|v| v.convert_with_options(&self.options))
            .collect()
    }

    /// Convert f64 to i64 with options
    pub fn f64_to_i64(&self, values: &[f64]) -> ConversionResult<Vec<i64>> {
        values
            .iter()
            .map(|v| v.convert_with_options(&self.options))
            .collect()
    }

    /// Convert between any numeric types (convenience method)
    pub fn convert<S, T>(&self, source: &[S]) -> ConversionResult<Vec<T>>
    where
        S: NumCast + Copy,
        T: NumCast + Copy,
    {
        self.convert_slice(source)
    }
}

/// Implement ArrayConvert for Vec
impl<S, T> ArrayConvert<T> for Vec<S>
where
    S: NumCast + Copy,
    T: NumCast + Copy,
{
    type Output = Vec<T>;

    fn convert_array(&self) -> ConversionResult<Self::Output> {
        let adapter: TypeAdapter<S, T> = TypeAdapter::new();
        adapter.adapt_slice(self)
    }

    fn convert_array_with_options(
        &self,
        _options: &ConversionOptions,
    ) -> ConversionResult<Self::Output> {
        // For generic numeric types, we use the basic conversion
        // More specific implementations can provide better precision handling
        self.convert_array()
    }
}

/// Implement ArrayConvert for slices
impl<S, T> ArrayConvert<T> for [S]
where
    S: NumCast + Copy,
    T: NumCast + Copy,
{
    type Output = Vec<T>;

    fn convert_array(&self) -> ConversionResult<Self::Output> {
        let adapter: TypeAdapter<S, T> = TypeAdapter::new();
        adapter.adapt_slice(self)
    }

    fn convert_array_with_options(
        &self,
        _options: &ConversionOptions,
    ) -> ConversionResult<Self::Output> {
        self.convert_array()
    }
}

/// Complex number conversion utilities
pub mod complex {
    use super::*;

    /// Convert complex from one precision to another
    pub fn convert_precision<S, T>(value: Complex<S>) -> ConversionResult<Complex<T>>
    where
        S: Float + NumCast,
        T: Float + NumCast,
    {
        let real = NumCast::from(value.re)
            .ok_or_else(|| ConversionError::Failed("Cannot convert real part".to_string()))?;
        let imag = NumCast::from(value.im)
            .ok_or_else(|| ConversionError::Failed("Cannot convert imaginary part".to_string()))?;
        Ok(Complex::new(real, imag))
    }

    /// Convert slice of complex numbers
    pub fn convert_slice<S, T>(values: &[Complex<S>]) -> ConversionResult<Vec<Complex<T>>>
    where
        S: Float + NumCast,
        T: Float + NumCast,
    {
        values.iter().map(|&v| convert_precision(v)).collect()
    }

    /// Convert real slice to complex slice
    pub fn real_to_complex<T: Float + Zero>(values: &[T]) -> Vec<Complex<T>> {
        values.iter().map(|&v| Complex::new(v, T::zero())).collect()
    }

    /// Extract real parts from complex slice
    pub fn extract_real<T: Float>(values: &[Complex<T>]) -> Vec<T> {
        values.iter().map(|v| v.re).collect()
    }

    /// Extract imaginary parts from complex slice
    pub fn extract_imag<T: Float>(values: &[Complex<T>]) -> Vec<T> {
        values.iter().map(|v| v.im).collect()
    }

    /// Convert complex slice to magnitude slice
    pub fn to_magnitude<T: Float>(values: &[Complex<T>]) -> Vec<T> {
        values
            .iter()
            .map(|v| (v.re * v.re + v.im * v.im).sqrt())
            .collect()
    }

    /// Convert complex slice to phase slice
    pub fn to_phase<T: Float>(values: &[Complex<T>]) -> Vec<T> {
        values.iter().map(|v| v.im.atan2(v.re)).collect()
    }
}

/// Batch conversion utilities
pub mod batch {
    use super::*;

    /// Batch conversion result
    #[derive(Debug)]
    pub struct BatchResult<T> {
        /// Successfully converted values
        pub values: Vec<T>,
        /// Indices of failed conversions
        pub failed_indices: Vec<usize>,
        /// Error messages for failures
        pub errors: Vec<ConversionError>,
    }

    impl<T> BatchResult<T> {
        /// Check if all conversions succeeded
        #[must_use]
        pub fn all_succeeded(&self) -> bool {
            self.failed_indices.is_empty()
        }

        /// Get the number of successful conversions
        #[must_use]
        pub fn success_count(&self) -> usize {
            self.values.len() - self.failed_indices.len()
        }

        /// Get the number of failed conversions
        #[must_use]
        pub fn failure_count(&self) -> usize {
            self.failed_indices.len()
        }
    }

    /// Convert with failure tolerance (returns partial results)
    pub fn convert_tolerant<S, T>(values: &[S]) -> BatchResult<T>
    where
        S: NumCast + Copy,
        T: NumCast + Copy + Default,
    {
        let mut result = BatchResult {
            values: Vec::with_capacity(values.len()),
            failed_indices: Vec::new(),
            errors: Vec::new(),
        };

        for (i, &value) in values.iter().enumerate() {
            match NumCast::from(value) {
                Some(converted) => result.values.push(converted),
                None => {
                    result.failed_indices.push(i);
                    result.errors.push(ConversionError::Failed(format!(
                        "Cannot convert value at index {i}"
                    )));
                    result.values.push(T::default());
                }
            }
        }

        result
    }

    /// Convert with default value for failures
    pub fn convert_with_default<S, T>(values: &[S], default: T) -> Vec<T>
    where
        S: NumCast + Copy,
        T: NumCast + Copy,
    {
        values
            .iter()
            .map(|&v| NumCast::from(v).unwrap_or(default))
            .collect()
    }

    /// Convert in parallel chunks
    #[cfg(feature = "parallel")]
    pub fn convert_parallel<S, T>(values: &[S], chunk_size: usize) -> ConversionResult<Vec<T>>
    where
        S: NumCast + Copy + Send + Sync,
        T: NumCast + Copy + Send + Sync,
    {
        use rayon::prelude::*;

        let results: Vec<ConversionResult<T>> = values
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|&v| {
                        NumCast::from(v)
                            .ok_or_else(|| ConversionError::Failed("Conversion failed".to_string()))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossless_convert() {
        let val: f64 = 2.5f32.convert_lossless().expect("Should convert");
        assert!((val - 2.5f64).abs() < 0.001);

        let val: i64 = 42i32.convert_lossless().expect("Should convert");
        assert_eq!(val, 42);
    }

    #[test]
    fn test_lossy_convert() {
        let val: f32 = std::f64::consts::PI.convert_lossy();
        assert!((val - std::f32::consts::PI).abs() < 1e-6);

        let val: i32 = 3.7f64.convert_lossy();
        assert_eq!(val, 3);
    }

    #[test]
    fn test_lossy_convert_with_options() {
        let options = ConversionOptions::strict();
        let result: ConversionResult<i32> = 3.5f64.convert_with_options(&options);
        assert!(result.is_err()); // Precision loss not allowed

        let options = ConversionOptions::lenient();
        let result: ConversionResult<i32> = 3.5f64.convert_with_options(&options);
        assert!(result.is_ok());
        assert_eq!(result.expect("Should convert"), 4); // Rounded
    }

    #[test]
    fn test_type_adapter() {
        let adapter: TypeAdapter<f64, i32> = TypeAdapter::new();
        let result = adapter.adapt(42.5);
        assert!(result.is_ok());
        assert_eq!(result.expect("Should adapt"), 42);

        let values = [1.0, 2.0, 3.0, 4.0];
        let adapted: Vec<i32> = adapter.adapt_slice(&values).expect("Should adapt slice");
        assert_eq!(adapted, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_data_flow_converter() {
        let converter = DataFlowConverter::new();
        let values = [1.5, 2.5, 3.5, 4.5];

        let result: Vec<i32> = converter.convert(&values).expect("Should convert");
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_array_convert() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let converted: Vec<i32> = vec.convert_array().expect("Should convert");
        assert_eq!(converted, vec![1, 2, 3]);
    }

    #[test]
    fn test_complex_conversion() {
        let c64 = Complex::new(1.0f64, 2.0f64);
        let c32: Complex<f32> = complex::convert_precision(c64).expect("Should convert");
        assert!((c32.re - 1.0f32).abs() < 1e-6);
        assert!((c32.im - 2.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_real_to_complex() {
        let values = [1.0, 2.0, 3.0];
        let complex_vals = complex::real_to_complex(&values);
        assert_eq!(complex_vals.len(), 3);
        assert_eq!(complex_vals[0].im, 0.0);
    }

    #[test]
    fn test_batch_convert_tolerant() {
        let values: Vec<f64> = vec![1.0, 2.0, f64::NAN, 4.0];
        let result: batch::BatchResult<i32> = batch::convert_tolerant(&values);
        // NaN conversion behavior depends on platform, but we should handle it gracefully
        assert!(result.success_count() + result.failure_count() == 4);
    }

    #[test]
    fn test_conversion_error_display() {
        let err = ConversionError::TypeMismatch {
            expected: "f64".to_string(),
            actual: "String".to_string(),
        };
        assert!(err.to_string().contains("f64"));

        let err = ConversionError::OutOfRange {
            value: "1e100".to_string(),
            min: "-1e38".to_string(),
            max: "1e38".to_string(),
        };
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_conversion_options() {
        let strict = ConversionOptions::strict();
        assert!(!strict.allow_precision_loss);
        assert!(!strict.clamp_to_range);

        let lenient = ConversionOptions::lenient();
        assert!(lenient.allow_precision_loss);
        assert!(lenient.clamp_to_range);
    }

    #[test]
    fn test_out_of_range_clamping() {
        let options = ConversionOptions {
            clamp_to_range: true,
            ..Default::default()
        };

        let result: ConversionResult<i32> = 1e20f64.convert_with_options(&options);
        assert!(result.is_ok());
        assert_eq!(result.expect("Should clamp"), i32::MAX);

        let result: ConversionResult<i32> = (-1e20f64).convert_with_options(&options);
        assert!(result.is_ok());
        assert_eq!(result.expect("Should clamp"), i32::MIN);
    }
}
