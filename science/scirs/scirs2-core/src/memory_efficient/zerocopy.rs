//! Zero-copy operations for memory-mapped arrays.
//!
//! This module provides a set of operations that can be performed on memory-mapped
//! arrays without loading the entire array into memory or making unnecessary copies.
//! These operations maintain memory-mapping where possible and only load the minimum
//! required data.

use super::chunked::ChunkingStrategy;
use super::memmap::{AccessMode, MemoryMappedArray};
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
// Use crate::ndarray for SciRS2 POLICY compliance (supports both ndarray 0.16 and 0.17)
use crate::ndarray;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};

/// Trait for zero-copy operations on memory-mapped arrays.
///
/// This trait provides methods for performing operations on memory-mapped arrays
/// without unnecessary memory allocations or copies. The operations are designed
/// to work efficiently with large datasets by processing data in chunks and
/// maintaining memory-mapping where possible.
pub trait ZeroCopyOps<A: Clone + Copy + 'static + Send + Sync> {
    /// Maps a function over each element of the array without loading the entire array.
    ///
    /// This is similar to the `map` function in functional programming, but implemented
    /// to work efficiently with memory-mapped arrays by processing chunks.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an element of type `A` and returns a new element of the same type
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the mapped values
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// // Double each element
    /// let doubled = mmap.map_zero_copy(|x| x * 2.0);
    /// ```
    fn map_zero_copy<F>(&self, f: F) -> CoreResult<MemoryMappedArray<A>>
    where
        F: Fn(A) -> A + Send + Sync;

    /// Reduces the array to a single value by applying a binary operation.
    ///
    /// This is similar to the `fold` or `reduce` function in functional programming,
    /// but implemented to work efficiently with memory-mapped arrays by processing chunks.
    ///
    /// # Arguments
    ///
    /// * `init` - The initial value for the reduction
    /// * `f` - A function that takes two values of type `A` and combines them into one
    ///
    /// # Returns
    ///
    /// The reduced value
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// // Sum all elements
    /// let sum = mmap.reduce_zero_copy(0.0, |acc, x| acc + x);
    /// ```
    fn reduce_zero_copy<F>(&self, init: A, f: F) -> CoreResult<A>
    where
        F: Fn(A, A) -> A + Send + Sync;

    /// Performs a binary operation between two memory-mapped arrays element-wise.
    ///
    /// This allows for operations like addition, subtraction, etc. between two arrays
    /// without loading both arrays entirely into memory.
    ///
    /// # Arguments
    ///
    /// * `other` - Another memory-mapped array with the same shape
    /// * `f` - A function that takes two elements (one from each array) and returns a new element
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the result of the binary operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap1: MemoryMappedArray<f64> = unimplemented!();
    /// # let mmap2: MemoryMappedArray<f64> = unimplemented!();
    /// // Add two arrays element-wise
    /// let sum_array = mmap1.combine_zero_copy(&mmap2, |a, b| a + b);
    /// ```
    fn combine_zero_copy<F>(&self, other: &Self, f: F) -> CoreResult<MemoryMappedArray<A>>
    where
        F: Fn(A, A) -> A + Send + Sync;

    /// Filters elements based on a predicate function.
    ///
    /// Returns a new array containing only the elements that satisfy the predicate.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that takes an element and returns a boolean
    ///
    /// # Returns
    ///
    /// A new array containing only the elements that satisfy the predicate
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// // Get only positive elements
    /// let positives = mmap.filter_zero_copy(|&x| x > 0.0);
    /// ```
    fn filter_zero_copy<F>(&self, predicate: F) -> CoreResult<Vec<A>>
    where
        F: Fn(&A) -> bool + Send + Sync;

    /// Returns the maximum element in the array.
    ///
    /// # Returns
    ///
    /// The maximum element, or an error if the array is empty
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// let max_value = mmap.max_zero_copy();
    /// ```
    fn max_zero_copy(&self) -> CoreResult<A>
    where
        A: PartialOrd;

    /// Returns the minimum element in the array.
    ///
    /// # Returns
    ///
    /// The minimum element, or an error if the array is empty
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// let min_value = mmap.min_zero_copy();
    /// ```
    fn min_zero_copy(&self) -> CoreResult<A>
    where
        A: PartialOrd;

    /// Calculates the sum of all elements in the array.
    ///
    /// # Returns
    ///
    /// The sum of all elements
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// let total = mmap.sum_zero_copy();
    /// ```
    fn sum_zero_copy(&self) -> CoreResult<A>
    where
        A: Add<Output = A> + From<u8>;

    /// Calculates the product of all elements in the array.
    ///
    /// # Returns
    ///
    /// The product of all elements
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// let product = mmap.product_zero_copy();
    /// ```
    fn product_zero_copy(&self) -> CoreResult<A>
    where
        A: Mul<Output = A> + From<u8>;

    /// Calculates the mean of all elements in the array.
    ///
    /// # Returns
    ///
    /// The mean of all elements
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, ZeroCopyOps};
    /// # let mmap: MemoryMappedArray<f64> = unimplemented!();
    /// let avg = mmap.mean_zero_copy();
    /// ```
    fn mean_zero_copy(&self) -> CoreResult<A>
    where
        A: Add<Output = A> + Div<Output = A> + From<u8> + From<usize>;
}

impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync + Zero> ZeroCopyOps<A>
    for MemoryMappedArray<A>
{
    fn map_zero_copy<F>(&self, f: F) -> CoreResult<MemoryMappedArray<A>>
    where
        F: Fn(A) -> A + Send + Sync,
    {
        // Create a temporary file for the result
        let temp_file = tempfile::NamedTempFile::new()?;
        let temp_path = temp_file.path().to_path_buf();

        // Create an output memory-mapped array with the same shape
        let shape = &self.shape;
        let element_size = std::mem::size_of::<A>();
        let file_size = self.size * element_size;

        // Resize the temp file to the required size
        temp_file.as_file().set_len(file_size as u64)?;
        drop(temp_file); // Close the file before memory-mapping it

        // Create the output memory-mapped array with zeros to initialize the header
        // First create with Write mode to initialize
        let zeros = crate::ndarray::ArrayD::zeros(crate::ndarray::IxDyn(&self.shape));
        {
            let _ = MemoryMappedArray::<A>::new::<
                crate::ndarray::OwnedRepr<A>,
                crate::ndarray::IxDyn,
            >(Some(&zeros), &temp_path, AccessMode::Write, 0)?;
        }

        // Now reopen in ReadWrite mode to allow modifications
        let mut output = MemoryMappedArray::<A>::new::<
            crate::ndarray::OwnedRepr<A>,
            crate::ndarray::IxDyn,
        >(None, &temp_path, AccessMode::ReadWrite, 0)?;

        // Process the input array in chunks
        #[cfg(feature = "parallel")]
        {
            // Use rayon directly since process_chunks_parallel has trait bounds
            // that we can't satisfy here (Send + Sync)

            let chunk_size = (self.size / rayon::current_num_threads()).max(1024);

            // Calculate the number of chunks
            let num_chunks = self.size.div_ceil(chunk_size);

            // Process each chunk sequentially to avoid mutable borrow issues
            let array = self.as_array::<crate::ndarray::IxDyn>()?;
            let mut out_array = output.as_array_mut::<crate::ndarray::IxDyn>()?;

            for chunk_idx in 0..num_chunks {
                // Calculate chunk bounds
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(self.size);

                // Get the data for this chunk
                let slice = array.as_slice().ok_or_else(|| {
                    CoreError::ValidationError(
                        ErrorContext::new("Array is not contiguous in memory".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                let chunk = &slice[start..end];

                // Apply the mapping function to each element in the chunk
                let mapped_chunk: Vec<A> = chunk.iter().map(|&x| f(x)).collect();

                // Copy the mapped chunk to the output at the same position
                let out_slice_full = out_array.as_slice_mut().ok_or_else(|| {
                    CoreError::ValidationError(
                        ErrorContext::new("Output array is not contiguous in memory".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                let out_slice = &mut out_slice_full[start..end];

                // Copy the mapped chunk to the output
                out_slice.copy_from_slice(&mapped_chunk);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Use sequential processing

            let chunk_size = 1024 * 1024; // 1M elements
            let strategy = ChunkingStrategy::Fixed(chunk_size);

            // Manually process chunks instead of using process_chunks_mut
            for chunk_idx in 0..self.size.div_ceil(chunk_size) {
                // Calculate chunk bounds
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(self.size);

                // Get the data for this chunk
                let array = self.as_array::<crate::ndarray::IxDyn>()?;
                let slice = array.as_slice().ok_or_else(|| {
                    CoreError::ValidationError(
                        ErrorContext::new("Array is not contiguous in memory".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                let chunk = &slice[start..end];

                // Apply the mapping function to each element in the chunk
                let mapped_chunk: Vec<A> = chunk.iter().map(|&x| f(x)).collect();

                // Copy the mapped chunk to the output at the same position
                // Get a mutable view of the output array
                let mut out_array = output.as_array_mut::<crate::ndarray::IxDyn>()?;
                let out_slice_full = out_array.as_slice_mut().ok_or_else(|| {
                    CoreError::ValidationError(
                        ErrorContext::new("Output array is not contiguous in memory".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                let out_slice = &mut out_slice_full[start..end];

                // Copy the mapped chunk to the output
                out_slice.copy_from_slice(&mapped_chunk);
            }
        }

        Ok(output)
    }

    fn reduce_zero_copy<F>(&self, init: A, f: F) -> CoreResult<A>
    where
        F: Fn(A, A) -> A + Send + Sync,
    {
        // Process the input array in chunks
        let chunk_size = 1024 * 1024; // 1M elements
        let strategy = ChunkingStrategy::Fixed(chunk_size);

        // Since we can't use process_chunks directly, we'll implement manually
        let num_chunks = self.size.div_ceil(chunk_size);
        let mut chunk_results = Vec::with_capacity(num_chunks);

        // Process each chunk
        for chunk_idx in 0..num_chunks {
            // Calculate chunk bounds
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(self.size);

            // Load the array
            let array = self.as_array::<crate::ndarray::IxDyn>()?;
            let slice = array.as_slice().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            let chunk = &slice[start..end];

            // Reduce the chunk
            let chunk_result = chunk.iter().fold(init, |acc, &x| f(acc, x));
            chunk_results.push(chunk_result);
        }

        // Combine chunk results
        let final_result = chunk_results.into_iter().fold(init, f);

        Ok(final_result)
    }

    fn combine_zero_copy<F>(&self, other: &Self, f: F) -> CoreResult<MemoryMappedArray<A>>
    where
        F: Fn(A, A) -> A + Send + Sync,
    {
        // Check that the arrays have the same shape
        if self.shape != other.shape {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Arrays have different shapes: {:?} vs {:?}",
                self.shape, other.shape
            ))));
        }

        // Create a temporary file for the result
        let temp_file = tempfile::NamedTempFile::new()?;
        let temp_path = temp_file.path().to_path_buf();

        // Create an output memory-mapped array with the same shape
        let shape = &self.shape;
        let element_size = std::mem::size_of::<A>();
        let file_size = self.size * element_size;

        // Resize the temp file to the required size
        temp_file.as_file().set_len(file_size as u64)?;
        drop(temp_file); // Close the file before memory-mapping it

        // Create the output memory-mapped array with zeros to initialize the header
        // First create with Write mode to initialize
        let zeros = crate::ndarray::ArrayD::zeros(crate::ndarray::IxDyn(&self.shape));
        {
            let _ = MemoryMappedArray::<A>::new::<
                crate::ndarray::OwnedRepr<A>,
                crate::ndarray::IxDyn,
            >(Some(&zeros), &temp_path, AccessMode::Write, 0)?;
        }

        // Now reopen in ReadWrite mode to allow modifications
        let mut output = MemoryMappedArray::<A>::new::<
            crate::ndarray::OwnedRepr<A>,
            crate::ndarray::IxDyn,
        >(None, &temp_path, AccessMode::ReadWrite, 0)?;

        // Process the arrays in chunks
        let chunk_size = 1024 * 1024; // 1M elements
        let strategy = ChunkingStrategy::Fixed(chunk_size);

        // Calculate the number of chunks
        let num_chunks = self.size.div_ceil(chunk_size);

        // Process each chunk
        for chunk_idx in 0..num_chunks {
            // Calculate chunk bounds
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(self.size);
            let len = end - start;

            // Load chunks from both arrays
            let self_array = self.as_array::<crate::ndarray::IxDyn>()?;
            let other_array = other.as_array::<crate::ndarray::IxDyn>()?;

            let self_slice = self_array.as_slice().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Self array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            let other_slice = other_array.as_slice().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Other array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            let self_chunk = &self_slice[start..end];
            let other_chunk = &other_slice[start..end];

            // Apply the binary operation
            let mut result_chunk = Vec::with_capacity(len);
            for i in 0..len {
                result_chunk.push(f(self_chunk[i], other_chunk[i]));
            }

            // Write the result to the output array
            let mut out_array = output.as_array_mut::<crate::ndarray::IxDyn>()?;
            let out_slice_full = out_array.as_slice_mut().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Output array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            let out_slice = &mut out_slice_full[start..end];
            out_slice.copy_from_slice(&result_chunk);
        }

        Ok(output)
    }

    fn filter_zero_copy<F>(&self, predicate: F) -> CoreResult<Vec<A>>
    where
        F: Fn(&A) -> bool + Send + Sync,
    {
        // Process the input array in chunks manually
        let chunk_size = 1024 * 1024; // 1M elements
        let num_chunks = self.size.div_ceil(chunk_size);
        let mut result = Vec::new();

        // Process each chunk
        for chunk_idx in 0..num_chunks {
            // Calculate chunk bounds
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(self.size);

            // Load the array
            let array = self.as_array::<crate::ndarray::IxDyn>()?;
            let array_slice = array.as_slice().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            let slice = &array_slice[start..end];

            // Filter the chunk
            let filtered_chunk = slice
                .iter()
                .filter(|&x| predicate(x))
                .cloned()
                .collect::<Vec<A>>();

            // Add filtered elements to the result
            result.extend(filtered_chunk);
        }

        Ok(result)
    }

    fn max_zero_copy(&self) -> CoreResult<A>
    where
        A: PartialOrd,
    {
        // Handle empty array
        if self.size == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Array is empty".to_string(),
            )));
        }

        // Read the first element to initialize
        let first_element = {
            let array = self.as_array::<crate::ndarray::IxDyn>()?;
            let slice = array.as_slice().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            slice[0]
        };

        // Reduce the array to find the maximum
        self.reduce_zero_copy(first_element, |acc, x| if x > acc { x } else { acc })
    }

    fn min_zero_copy(&self) -> CoreResult<A>
    where
        A: PartialOrd,
    {
        // Handle empty array
        if self.size == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Array is empty".to_string(),
            )));
        }

        // Read the first element to initialize
        let first_element = {
            let array = self.as_array::<crate::ndarray::IxDyn>()?;
            let slice = array.as_slice().ok_or_else(|| {
                CoreError::ValidationError(
                    ErrorContext::new("Array is not contiguous in memory".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            slice[0]
        };

        // Reduce the array to find the minimum
        self.reduce_zero_copy(first_element, |acc, x| if x < acc { x } else { acc })
    }

    fn sum_zero_copy(&self) -> CoreResult<A>
    where
        A: Add<Output = A> + From<u8>,
    {
        // Initialize with zero
        let zero = A::from(0u8);

        // Sum all elements
        self.reduce_zero_copy(zero, |acc, x| acc + x)
    }

    fn product_zero_copy(&self) -> CoreResult<A>
    where
        A: Mul<Output = A> + From<u8>,
    {
        // Handle empty array
        if self.size == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Array is empty".to_string(),
            )));
        }

        // Initialize with one
        let one = A::from(1u8);

        // Multiply all elements
        self.reduce_zero_copy(one, |acc, x| acc * x)
    }

    fn mean_zero_copy(&self) -> CoreResult<A>
    where
        A: Add<Output = A> + Div<Output = A> + From<u8> + From<usize>,
    {
        // Handle empty array
        if self.size == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Array is empty".to_string(),
            )));
        }

        // Calculate sum
        let sum = self.sum_zero_copy()?;

        // Divide by count
        let count = A::from(self.size);

        Ok(sum / count)
    }
}

/// Trait for broadcasting operations between memory-mapped arrays of different shapes.
///
/// This trait provides methods for performing broadcasting operations between
/// memory-mapped arrays without unnecessary memory allocations or copies.
pub trait BroadcastOps<A: Clone + Copy + 'static + Send + Sync> {
    /// Broadcasts an operation between two arrays of compatible shapes.
    ///
    /// Follows the `NumPy` broadcasting rules:
    /// 1. If arrays don't have the same rank, prepend shape with 1s
    /// 2. Two dimensions are compatible if:
    ///    - They are equal, or
    ///    - One of them is 1
    ///
    /// # Arguments
    ///
    /// * `other` - Another memory-mapped array with a compatible shape
    /// * `f` - A function that takes two elements (one from each array) and returns a new element
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the result of the broadcasted operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use scirs2_core::memory_efficient::{MemoryMappedArray, BroadcastOps};
    /// # let mmap1: MemoryMappedArray<f64> = unimplemented!(); // Shape [3, 4]
    /// # let mmap2: MemoryMappedArray<f64> = unimplemented!(); // Shape [4]
    /// // Broadcast and multiply
    /// let result = mmap1.broadcast_op(&mmap2, |a, b| a * b);
    /// ```
    fn broadcast_op<F>(&self, other: &Self, f: F) -> CoreResult<MemoryMappedArray<A>>
    where
        F: Fn(A, A) -> A + Send + Sync;
}

impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync + Zero> BroadcastOps<A>
    for MemoryMappedArray<A>
{
    fn broadcast_op<F>(&self, other: &Self, f: F) -> CoreResult<MemoryMappedArray<A>>
    where
        F: Fn(A, A) -> A + Send + Sync,
    {
        // Check shape compatibility for broadcasting
        let selfshape = &self.shape;
        let othershape = &other.shape;

        // Get the dimensions
        let self_ndim = selfshape.len();
        let other_ndim = othershape.len();
        let output_ndim = std::cmp::max(self_ndim, other_ndim);

        // Convert shapes to vectors with leading 1s as needed
        let mut self_dims = Vec::with_capacity(output_ndim);
        let mut other_dims = Vec::with_capacity(output_ndim);

        // Prepend 1s to the shape with fewer dimensions
        self_dims.resize(output_ndim - self_ndim, 1);
        for dim in selfshape.iter() {
            self_dims.push(*dim);
        }

        other_dims.resize(output_ndim - other_ndim, 1);
        for dim in othershape.iter() {
            other_dims.push(*dim);
        }

        // Determine the output shape
        let mut outputshape = Vec::with_capacity(output_ndim);
        for i in 0..output_ndim {
            #[allow(clippy::if_same_then_else)]
            if self_dims[i] == 1 {
                outputshape.push(other_dims[i]);
            } else if other_dims[i] == 1 {
                outputshape.push(self_dims[i]);
            } else if self_dims[i] == other_dims[i] {
                outputshape.push(self_dims[i]);
            } else {
                return Err(CoreError::ValueError(ErrorContext::new(format!(
                    "Arrays cannot be broadcast together with shapes {selfshape:?} and {othershape:?}"
                ))));
            }
        }

        // Create a temporary file for the result
        let temp_file = tempfile::NamedTempFile::new()?;
        let temp_path = temp_file.path().to_path_buf();

        // Calculate the output array size
        let output_size = outputshape.iter().product::<usize>();
        let element_size = std::mem::size_of::<A>();
        let file_size = output_size * element_size;

        // Resize the temp file to the required size
        temp_file.as_file().set_len(file_size as u64)?;
        drop(temp_file); // Close the file before memory-mapping it

        // Create the output memory-mapped array with zeros to initialize the header
        // Use the calculated output shape instead of self.shape
        let zeros = crate::ndarray::ArrayD::zeros(crate::ndarray::IxDyn(&outputshape));
        {
            let _ = MemoryMappedArray::<A>::new::<
                crate::ndarray::OwnedRepr<A>,
                crate::ndarray::IxDyn,
            >(Some(&zeros), &temp_path, AccessMode::Write, 0)?;
        }

        // Now reopen in ReadWrite mode to allow modifications
        let mut output = MemoryMappedArray::<A>::new::<
            crate::ndarray::OwnedRepr<A>,
            crate::ndarray::IxDyn,
        >(None, &temp_path, AccessMode::ReadWrite, 0)?;

        // Load both arrays into memory (for broadcasting, we need random access)
        let self_array = self.as_array::<crate::ndarray::IxDyn>()?;
        let other_array = other.as_array::<crate::ndarray::IxDyn>()?;

        // Create ndarray views for easier broadcasting
        let self_view = self_array.view();
        let other_view = other_array.view();

        // Perform the broadcasted operation
        let mut output_array = output.as_array_mut::<crate::ndarray::IxDyn>()?;

        // Use ndarray's broadcasting capability
        crate::ndarray::Zip::from(&mut output_array)
            .and_broadcast(&self_view)
            .and_broadcast(&other_view)
            .for_each(|out, &a, &b| {
                *out = f(a, b);
            });

        Ok(output)
    }
}

/// Extension trait for standard arithmetic operations on memory-mapped arrays.
///
/// This trait provides implementations of standard arithmetic operations
/// (addition, subtraction, multiplication, division) for memory-mapped arrays
/// using the zero-copy infrastructure.
pub trait ArithmeticOps<A: Clone + Copy + 'static + Send + Sync> {
    /// Adds two arrays element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - Another memory-mapped array with the same shape
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the sum
    fn add(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Add<Output = A>;

    /// Subtracts another array from this one element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - Another memory-mapped array with the same shape
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the difference
    fn sub(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Sub<Output = A>;

    /// Multiplies two arrays element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - Another memory-mapped array with the same shape
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the product
    fn mul(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Mul<Output = A>;

    /// Divides this array by another element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - Another memory-mapped array with the same shape
    ///
    /// # Returns
    ///
    /// A new memory-mapped array containing the quotient
    fn div(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Div<Output = A>;
}

impl<A: Clone + Copy + 'static + Send + Sync + Send + Sync + Zero> ArithmeticOps<A>
    for MemoryMappedArray<A>
{
    fn add(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Add<Output = A>,
    {
        self.combine_zero_copy(other, |a, b| a + b)
    }

    fn sub(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Sub<Output = A>,
    {
        self.combine_zero_copy(other, |a, b| a - b)
    }

    fn mul(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Mul<Output = A>,
    {
        self.combine_zero_copy(other, |a, b| a * b)
    }

    fn div(&self, other: &Self) -> CoreResult<MemoryMappedArray<A>>
    where
        A: Div<Output = A>,
    {
        self.combine_zero_copy(other, |a, b| a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::Array2;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_map_zero_copy() {
        // Create a temporary directory for our test files
        let dir = tempdir().expect("Operation failed");
        let file_path = dir.path().join("test_map.bin");

        // Create a test array and save it with proper header using save_array
        let data = crate::ndarray::Array1::from_vec((0..1000).map(|i| i as f64).collect());
        MemoryMappedArray::<f64>::save_array(&data, &file_path, None).expect("Operation failed");

        // Open the file for zero-copy operations
        let mmap = MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly)
            .expect("Operation failed");

        // Map operation: double each element
        let result = mmap.map_zero_copy(|x| x * 2.0).expect("Operation failed");

        // Verify the result
        let result_array = result
            .readonlyarray::<crate::ndarray::Ix1>()
            .expect("Operation failed");
        for i in 0..1000 {
            assert_eq!(result_array[i], (i as f64) * 2.0);
        }
    }

    #[test]
    fn test_reduce_zero_copy() {
        // Create a temporary directory for our test files
        let dir = tempdir().expect("Operation failed");
        let file_path = dir.path().join("test_reduce.bin");

        // Create a test array and save it to a file
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).expect("Operation failed");
        for val in &data {
            file.write_all(&val.to_ne_bytes())
                .expect("Operation failed");
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::path(&file_path, &[1000]).expect("Operation failed");

        // Reduce operation: sum all elements
        let sum = mmap
            .reduce_zero_copy(0.0, |acc, x| acc + x)
            .expect("Operation failed");

        // Verify the result (sum of 0..999 = 499500)
        assert_eq!(sum, 499500.0);
    }

    #[test]
    fn test_combine_zero_copy() {
        // Create a temporary directory for our test files
        let dir = tempdir().expect("Operation failed");
        let file_path1 = dir.path().join("test_combine1.bin");
        let file_path2 = dir.path().join("test_combine2.bin");

        // Create two test arrays and save them with proper headers using save_array
        let data1 = crate::ndarray::Array1::from_vec((0..1000).map(|i| i as f64).collect());
        let data2 = crate::ndarray::Array1::from_vec((0..1000).map(|i| (i * 2) as f64).collect());

        MemoryMappedArray::<f64>::save_array(&data1, &file_path1, None).expect("Operation failed");
        MemoryMappedArray::<f64>::save_array(&data2, &file_path2, None).expect("Operation failed");

        // Open the files for zero-copy operations
        let mmap1 = MemoryMappedArray::<f64>::open_zero_copy(&file_path1, AccessMode::ReadOnly)
            .expect("Operation failed");
        let mmap2 = MemoryMappedArray::<f64>::open_zero_copy(&file_path2, AccessMode::ReadOnly)
            .expect("Operation failed");

        // Combine operation: add the arrays
        let result = mmap1
            .combine_zero_copy(&mmap2, |a, b| a + b)
            .expect("Operation failed");

        // Verify the result (each element should be 3*i)
        let result_array = result
            .readonlyarray::<crate::ndarray::Ix1>()
            .expect("Operation failed");
        for i in 0..1000 {
            assert_eq!(result_array[i], (i as f64) * 3.0);
        }
    }

    #[test]
    fn test_filter_zero_copy() {
        // Create a temporary directory for our test files
        let dir = tempdir().expect("Operation failed");
        let file_path = dir.path().join("test_filter.bin");

        // Create a test array and save it to a file
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let mut file = File::create(&file_path).expect("Operation failed");
        for val in &data {
            file.write_all(&val.to_ne_bytes())
                .expect("Operation failed");
        }
        drop(file);

        // Create a memory-mapped array
        let mmap = MemoryMappedArray::<f64>::path(&file_path, &[1000]).expect("Operation failed");

        // Filter operation: keep only even numbers
        let even_numbers = mmap
            .filter_zero_copy(|&x| (x as usize) % 2 == 0)
            .expect("Operation failed");

        // Verify the result (should be 0, 2, 4, ..., 998)
        assert_eq!(even_numbers.len(), 500);
        for (i, val) in even_numbers.iter().enumerate() {
            assert_eq!(*val, (i * 2) as f64);
        }
    }

    #[test]
    fn test_arithmetic_ops() {
        // Create a temporary directory for our test files
        let dir = tempdir().expect("Operation failed");
        let file_path1 = dir.path().join("test_arithmetic1.bin");
        let file_path2 = dir.path().join("test_arithmetic2.bin");

        // Create two test arrays and save them with proper headers using save_array
        let data1 = crate::ndarray::Array1::from_vec((0..100).map(|i| i as f64).collect());
        let data2 = crate::ndarray::Array1::from_vec((0..100).map(|i| (i + 5) as f64).collect());

        MemoryMappedArray::<f64>::save_array(&data1, &file_path1, None).expect("Operation failed");
        MemoryMappedArray::<f64>::save_array(&data2, &file_path2, None).expect("Operation failed");

        // Open the files for zero-copy operations
        let mmap1 = MemoryMappedArray::<f64>::open_zero_copy(&file_path1, AccessMode::ReadOnly)
            .expect("Operation failed");
        let mmap2 = MemoryMappedArray::<f64>::open_zero_copy(&file_path2, AccessMode::ReadOnly)
            .expect("Operation failed");

        // Test addition
        let add_result = mmap1.add(&mmap2).expect("Operation failed");
        let add_array = add_result
            .readonlyarray::<crate::ndarray::Ix1>()
            .expect("Operation failed");
        for i in 0..100 {
            assert_eq!(add_array[i], (i as f64) + ((i + 5) as f64));
        }

        // Test subtraction
        let sub_result = mmap1.sub(&mmap2).expect("Operation failed");
        let sub_array = sub_result
            .readonlyarray::<crate::ndarray::Ix1>()
            .expect("Operation failed");
        for i in 0..100 {
            assert_eq!(sub_array[i], (i as f64) - ((i + 5) as f64));
        }

        // Test multiplication
        let mul_result = mmap1.mul(&mmap2).expect("Operation failed");
        let mul_array = mul_result
            .readonlyarray::<crate::ndarray::Ix1>()
            .expect("Operation failed");
        for i in 0..100 {
            assert_eq!(mul_array[i], (i as f64) * ((i + 5) as f64));
        }

        // Test division (avoid division by zero)
        let div_result = mmap2
            .div(&mmap1.map_zero_copy(|x| x + 1.0).expect("Operation failed"))
            .expect("Test: operation failed");
        let div_array = div_result
            .readonlyarray::<crate::ndarray::Ix1>()
            .expect("Operation failed");
        for i in 0..100 {
            assert_eq!(div_array[i], ((i + 5) as f64) / ((i + 1) as f64));
        }
    }

    #[test]
    fn test_broadcast_op() {
        // Create a temporary directory for our test files
        let dir = tempdir().expect("Operation failed");
        let file_path1 = dir.path().join("test_broadcast1.bin");
        let file_path2 = dir.path().join("test_broadcast2.bin");

        // Create a 2D array (3x4) and a 1D array (4)
        let data1 = Array2::<f64>::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let data2 = crate::ndarray::Array1::from_vec((0..4).map(|i| (i + 1) as f64).collect());

        // Save the arrays with proper headers using save_array
        MemoryMappedArray::<f64>::save_array(&data1, &file_path1, None).expect("Operation failed");
        MemoryMappedArray::<f64>::save_array(&data2, &file_path2, None).expect("Operation failed");

        // Open the files for zero-copy operations
        let mmap1 = MemoryMappedArray::<f64>::open_zero_copy(&file_path1, AccessMode::ReadOnly)
            .expect("Operation failed");
        let mmap2 = MemoryMappedArray::<f64>::open_zero_copy(&file_path2, AccessMode::ReadOnly)
            .expect("Operation failed");

        // Test broadcasting
        let result = mmap1
            .broadcast_op(&mmap2, |a, b| a * b)
            .expect("Operation failed");

        // Verify the result
        let result_array = result
            .readonlyarray::<crate::ndarray::Ix2>()
            .expect("Operation failed");
        assert_eq!(result_array.shape(), &[3, 4]);

        for i in 0..3 {
            for j in 0..4 {
                let expected = (i * 4 + j) as f64 * (j + 1) as f64;
                assert_eq!(result_array[[i, j]], expected);
            }
        }
    }
}
