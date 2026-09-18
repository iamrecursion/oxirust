//! Embedded Systems Support for SciRS2 (v0.2.0)
//!
//! This module provides no_std-compatible utilities and abstractions for embedded systems,
//! including ARM Cortex-M, RISC-V, AVR, and embedded Linux targets.
//!
//! # Features
//!
//! - **no_std Compatible**: Works without standard library
//! - **Fixed-Point Arithmetic**: For systems without FPU
//! - **Stack-Based Operations**: Minimal heap usage
//! - **Deterministic Execution**: Predictable timing for real-time systems
//! - **Memory-Constrained**: Optimized for limited RAM/ROM
//!
//! # Example: Basic Embedded Usage
//!
//! ```rust,ignore
//! #![no_std]
//! #![no_main]
//!
//! use scirs2_core::embedded::*;
//!
//! #[entry]
//! fn main() -> ! {
//!     // Fixed-size array operations (no heap allocation)
//!     let data = StackArray::<f32, 16>::new();
//!
//!     // Deterministic computation
//!     let result = stack_based_filter(&data);
//!
//!     loop {}
//! }
//! ```
//!
//! # Memory Requirements
//!
//! - **Minimum RAM**: 8KB (basic operations)
//! - **Recommended RAM**: 32KB (full functionality)
//! - **Flash/ROM**: 64KB (core functionality)
//!
//! # Supported Platforms
//!
//! - ARM Cortex-M (M0/M0+/M3/M4/M7/M33)
//! - RISC-V (RV32I/RV32IMC/RV32IMAC)
//! - AVR (ATmega series)
//! - Embedded Linux (no libc)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use crate::error::{CoreError, CoreResult};
use core::marker::PhantomData;
use num_traits::{Float, NumAssignOps, Zero};

// Re-exports for convenience
pub use error_handling::{EmbeddedError, EmbeddedResult};
pub use memory_estimation::{estimate_stack_usage, MemoryRequirement};
pub use stack_array::{BufferFullError, FixedSizeBuffer, StackArray};

/// Stack-based computation for element-wise operations
///
/// Performs operations without heap allocation, suitable for embedded systems.
///
/// # Example
///
/// ```rust
/// use scirs2_core::embedded::StackArray;
///
/// let a = StackArray::<f32, 8>::new();
/// let b = StackArray::<f32, 8>::new();
/// // Perform element-wise operations
/// ```
#[inline]
pub fn stack_add<T, const N: usize>(a: &StackArray<T, N>, b: &StackArray<T, N>) -> StackArray<T, N>
where
    T: Float + NumAssignOps + Copy + Default,
{
    let mut result = StackArray::new();
    for i in 0..N {
        result.data[i] = a.data[i] + b.data[i];
    }
    result
}

/// Stack-based multiplication
#[inline]
pub fn stack_mul<T, const N: usize>(a: &StackArray<T, N>, b: &StackArray<T, N>) -> StackArray<T, N>
where
    T: Float + NumAssignOps + Copy + Default,
{
    let mut result = StackArray::new();
    for i in 0..N {
        result.data[i] = a.data[i] * b.data[i];
    }
    result
}

/// Calculate sum without heap allocation
#[inline]
pub fn stack_sum<T, const N: usize>(arr: &StackArray<T, N>) -> T
where
    T: Float + NumAssignOps + Copy + Default + Zero,
{
    let mut sum = T::zero();
    for i in 0..N {
        sum += arr.data[i];
    }
    sum
}

/// Calculate mean without heap allocation
#[inline]
pub fn stack_mean<T, const N: usize>(arr: &StackArray<T, N>) -> T
where
    T: Float + NumAssignOps + Copy + Default + Zero,
{
    if N == 0 {
        return T::zero();
    }
    stack_sum(arr) / T::from(N).expect("Failed to convert N to T")
}

/// Memory estimation utilities
pub mod memory_estimation {
    use core::mem;

    /// Estimated memory requirement for an operation
    #[derive(Debug, Copy, Clone)]
    pub struct MemoryRequirement {
        /// Stack memory in bytes
        pub stack_bytes: usize,
        /// Heap memory in bytes (if alloc feature enabled)
        pub heap_bytes: usize,
        /// ROM/Flash memory in bytes
        pub flash_bytes: usize,
    }

    impl MemoryRequirement {
        /// Create a new memory requirement estimate
        pub const fn new(stack_bytes: usize, heap_bytes: usize, flash_bytes: usize) -> Self {
            Self {
                stack_bytes,
                heap_bytes,
                flash_bytes,
            }
        }

        /// Estimate for basic operations
        pub const fn basic() -> Self {
            Self::new(1024, 0, 4096)
        }

        /// Estimate for signal processing
        pub const fn signal_processing() -> Self {
            Self::new(4096, 0, 16384)
        }

        /// Estimate for linear algebra
        pub const fn linalg() -> Self {
            Self::new(8192, 0, 32768)
        }
    }

    /// Estimate stack usage for a given array size
    pub const fn estimate_stack_usage<T>(count: usize) -> usize {
        mem::size_of::<T>() * count
    }
}

/// Error handling for embedded systems
pub mod error_handling {
    use core::fmt;

    /// Embedded-specific error types (no heap allocation)
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum EmbeddedError {
        /// Buffer overflow
        BufferOverflow,
        /// Buffer underflow
        BufferUnderflow,
        /// Invalid size
        InvalidSize,
        /// Out of memory
        OutOfMemory,
        /// Numerical error
        NumericalError,
        /// Not supported in no_std mode
        NotSupported,
    }

    impl fmt::Display for EmbeddedError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                EmbeddedError::BufferOverflow => write!(f, "Buffer overflow"),
                EmbeddedError::BufferUnderflow => write!(f, "Buffer underflow"),
                EmbeddedError::InvalidSize => write!(f, "Invalid size"),
                EmbeddedError::OutOfMemory => write!(f, "Out of memory"),
                EmbeddedError::NumericalError => write!(f, "Numerical error"),
                EmbeddedError::NotSupported => write!(f, "Not supported in no_std mode"),
            }
        }
    }

    /// Result type for embedded operations
    pub type EmbeddedResult<T> = Result<T, EmbeddedError>;
}

/// Stack array module
pub mod stack_array {
    use core::marker::PhantomData;

    /// Error returned when attempting to push to a full buffer
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BufferFullError;

    impl core::fmt::Display for BufferFullError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "buffer is full")
        }
    }

    /// Stack-based array for no-heap operations
    ///
    /// This provides a fixed-size array allocated on the stack,
    /// suitable for embedded systems without heap allocation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::embedded::StackArray;
    ///
    /// let mut data = StackArray::<f32, 16>::new();
    /// data[0] = 1.0;
    /// data[1] = 2.0;
    /// assert_eq!(data.len(), 16);
    /// ```
    #[derive(Debug, Clone)]
    pub struct StackArray<T, const N: usize> {
        pub(super) data: [T; N],
        _marker: PhantomData<T>,
    }

    impl<T: Copy + Default, const N: usize> StackArray<T, N> {
        /// Create a new stack-allocated array initialized with default values
        #[inline]
        pub fn new() -> Self {
            Self {
                data: [T::default(); N],
                _marker: PhantomData,
            }
        }

        /// Create from a slice (copies data if it fits)
        #[inline]
        pub fn from_slice(slice: &[T]) -> Option<Self> {
            if slice.len() != N {
                return None;
            }
            let mut result = Self::new();
            result.data.copy_from_slice(slice);
            Some(result)
        }

        /// Get the length of the array
        #[inline]
        pub const fn len(&self) -> usize {
            N
        }

        /// Check if the array is empty (always false for const generic arrays)
        #[inline]
        pub const fn is_empty(&self) -> bool {
            N == 0
        }

        /// Get a reference to the underlying array
        #[inline]
        pub fn as_slice(&self) -> &[T] {
            &self.data
        }

        /// Get a mutable reference to the underlying array
        #[inline]
        pub fn as_mut_slice(&mut self) -> &mut [T] {
            &mut self.data
        }
    }

    impl<T: Copy + Default, const N: usize> Default for StackArray<T, N> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T, const N: usize> core::ops::Index<usize> for StackArray<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, index: usize) -> &Self::Output {
            &self.data[index]
        }
    }

    impl<T, const N: usize> core::ops::IndexMut<usize> for StackArray<T, N> {
        #[inline]
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[index]
        }
    }

    /// Fixed-size buffer for streaming operations
    ///
    /// Useful for real-time signal processing where data arrives continuously.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::embedded::FixedSizeBuffer;
    ///
    /// let mut buffer = FixedSizeBuffer::<f32, 8>::new();
    /// buffer.push(1.0).expect("Buffer should not be full");
    /// buffer.push(2.0).expect("Buffer should not be full");
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[derive(Debug, Clone)]
    pub struct FixedSizeBuffer<T, const N: usize> {
        pub(super) data: [T; N],
        len: usize,
        _marker: PhantomData<T>,
    }

    impl<T: Copy + Default, const N: usize> FixedSizeBuffer<T, N> {
        /// Create a new empty buffer
        #[inline]
        pub fn new() -> Self {
            Self {
                data: [T::default(); N],
                len: 0,
                _marker: PhantomData,
            }
        }

        /// Push an element to the buffer (returns error if full)
        #[inline]
        pub fn push(&mut self, value: T) -> Result<(), BufferFullError> {
            if self.len >= N {
                return Err(BufferFullError);
            }
            self.data[self.len] = value;
            self.len += 1;
            Ok(())
        }

        /// Pop an element from the buffer (returns None if empty)
        #[inline]
        pub fn pop(&mut self) -> Option<T> {
            if self.len == 0 {
                return None;
            }
            self.len -= 1;
            Some(self.data[self.len])
        }

        /// Get the current length
        #[inline]
        pub const fn len(&self) -> usize {
            self.len
        }

        /// Check if the buffer is empty
        #[inline]
        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Check if the buffer is full
        #[inline]
        pub const fn is_full(&self) -> bool {
            self.len == N
        }

        /// Get the capacity
        #[inline]
        pub const fn capacity(&self) -> usize {
            N
        }

        /// Clear the buffer
        #[inline]
        pub fn clear(&mut self) {
            self.len = 0;
        }

        /// Get a slice of the valid data
        #[inline]
        pub fn as_slice(&self) -> &[T] {
            &self.data[..self.len]
        }
    }

    impl<T: Copy + Default, const N: usize> Default for FixedSizeBuffer<T, N> {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_array_basic() {
        let mut arr = StackArray::<f32, 8>::new();
        arr[0] = 1.0;
        arr[1] = 2.0;
        assert_eq!(arr.len(), 8);
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.0);
    }

    #[test]
    fn test_fixed_size_buffer() {
        let mut buffer = FixedSizeBuffer::<f32, 4>::new();
        assert!(buffer.is_empty());

        buffer.push(1.0).expect("Failed to push");
        buffer.push(2.0).expect("Failed to push");
        assert_eq!(buffer.len(), 2);

        assert_eq!(buffer.pop(), Some(2.0));
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_stack_operations() {
        let mut a = StackArray::<f32, 4>::new();
        let mut b = StackArray::<f32, 4>::new();

        for i in 0..4 {
            a[i] = i as f32;
            b[i] = (i + 1) as f32;
        }

        let sum = stack_add(&a, &b);
        assert_eq!(sum[0], 1.0);
        assert_eq!(sum[1], 3.0);

        let product = stack_mul(&a, &b);
        assert_eq!(product[0], 0.0);
        assert_eq!(product[1], 2.0);
    }

    #[test]
    fn test_stack_statistics() {
        let mut arr = StackArray::<f32, 4>::new();
        arr[0] = 1.0;
        arr[1] = 2.0;
        arr[2] = 3.0;
        arr[3] = 4.0;

        let total = stack_sum(&arr);
        assert_eq!(total, 10.0);

        let avg = stack_mean(&arr);
        assert_eq!(avg, 2.5);
    }
}
