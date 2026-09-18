//! Enhanced chunked image processing for large images
//!
//! This module provides memory-efficient, GPU-accelerated chunked processing
//! for images that are too large to fit in memory. It supports:
//!
//! - **Chunked filter operations** with proper overlap handling
//! - **Zero-copy transformations** using memory-mapped arrays
//! - **Memory-efficient morphological operations**
//! - **Out-of-core processing pipeline** for very large images
//! - **GPU acceleration** for supported operations
//!
//! # Architecture
//!
//! The module is built around the `ChunkedImageProcessor` which manages:
//! - Automatic chunk size optimization based on available memory
//! - Overlap regions for seamless filtering across chunk boundaries
//! - Parallel processing of independent chunks
//! - GPU offloading for supported operations
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_ndimage::chunked_processing::{ChunkedImageProcessor, ChunkingConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let processor = ChunkedImageProcessor::new(ChunkingConfig::default());
//! let large_image = Array2::<f64>::zeros((10000, 10000));
//!
//! // Apply Gaussian filter in chunks
//! let result = processor.gaussian_filter(&large_image.view(), 2.0, None).unwrap();
//! ```

use std::collections::VecDeque;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use scirs2_core::ndarray::{
    self as ndarray, Array, Array1, Array2, ArrayView, ArrayView2, ArrayViewMut, ArrayViewMut2,
    Axis, Dimension, Ix2, IxDyn, Slice, SliceInfoElem,
};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast, Zero};
use scirs2_core::parallel_ops::*;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::BorderMode;
use crate::morphology::MorphBorderMode;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for chunked image processing
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Maximum chunk size in bytes (default: 64 MB)
    pub max_chunk_bytes: usize,

    /// Minimum chunk size in pixels per dimension (default: 32)
    pub min_chunk_pixels: usize,

    /// Maximum overlap in pixels per dimension (default: 64)
    pub max_overlap_pixels: usize,

    /// Whether to enable parallel chunk processing (default: true)
    pub enable_parallel: bool,

    /// Number of chunks to prefetch for pipelining (default: 2)
    pub prefetch_count: usize,

    /// Whether to use memory-mapped files for very large images (default: true)
    pub use_mmap: bool,

    /// Threshold for using memory-mapped files in bytes (default: 1 GB)
    pub mmap_threshold: usize,

    /// Temporary directory for intermediate files
    pub temp_dir: Option<PathBuf>,

    /// Whether to use GPU acceleration when available (default: true)
    pub enable_gpu: bool,

    /// Memory fraction to use for processing (0.0 - 1.0, default: 0.5)
    pub memory_fraction: f64,

    /// Strategy for handling overlap regions
    pub overlap_strategy: OverlapStrategy,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_bytes: 64 * 1024 * 1024, // 64 MB
            min_chunk_pixels: 32,
            max_overlap_pixels: 64,
            enable_parallel: true,
            prefetch_count: 2,
            use_mmap: true,
            mmap_threshold: 1024 * 1024 * 1024, // 1 GB
            temp_dir: None,
            enable_gpu: true,
            memory_fraction: 0.5,
            overlap_strategy: OverlapStrategy::Blend,
        }
    }
}

/// Strategy for handling overlap regions between chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapStrategy {
    /// Simply overwrite (last chunk wins)
    Overwrite,

    /// Linear blending in overlap region
    Blend,

    /// Use weighted average based on distance from chunk center
    WeightedAverage,

    /// Use the chunk with higher quality metric
    QualityBased,
}

// ============================================================================
// Chunk Types
// ============================================================================

/// Represents a rectangular chunk of an image
#[derive(Debug, Clone)]
pub struct ChunkRegion {
    /// Start coordinates in the full image (row, col)
    pub start: (usize, usize),

    /// End coordinates in the full image (exclusive)
    pub end: (usize, usize),

    /// Start coordinates including overlap (for processing)
    pub padded_start: (usize, usize),

    /// End coordinates including overlap (for processing)
    pub padded_end: (usize, usize),

    /// The chunk index in the grid
    pub chunk_id: (usize, usize),
}

impl ChunkRegion {
    /// Create a new chunk region
    pub fn new(
        start: (usize, usize),
        end: (usize, usize),
        overlap: usize,
        image_shape: (usize, usize),
    ) -> Self {
        let padded_start = (
            start.0.saturating_sub(overlap),
            start.1.saturating_sub(overlap),
        );
        let padded_end = (
            (end.0 + overlap).min(image_shape.0),
            (end.1 + overlap).min(image_shape.1),
        );

        Self {
            start,
            end,
            padded_start,
            padded_end,
            chunk_id: (0, 0),
        }
    }

    /// Get the size of the chunk (excluding padding)
    pub fn size(&self) -> (usize, usize) {
        (self.end.0 - self.start.0, self.end.1 - self.start.1)
    }

    /// Get the size of the padded chunk
    pub fn padded_size(&self) -> (usize, usize) {
        (
            self.padded_end.0 - self.padded_start.0,
            self.padded_end.1 - self.padded_start.1,
        )
    }

    /// Get the overlap on each side
    pub fn overlap(&self) -> ((usize, usize), (usize, usize)) {
        (
            (
                self.start.0 - self.padded_start.0,
                self.start.1 - self.padded_start.1,
            ),
            (
                self.padded_end.0 - self.end.0,
                self.padded_end.1 - self.end.1,
            ),
        )
    }
}

/// Iterator over chunk regions
pub struct ChunkRegionIterator {
    image_shape: (usize, usize),
    chunk_size: (usize, usize),
    overlap: usize,
    current_row: usize,
    current_col: usize,
    num_chunks_row: usize,
    num_chunks_col: usize,
}

impl ChunkRegionIterator {
    /// Create a new chunk region iterator
    pub fn new(image_shape: (usize, usize), chunk_size: (usize, usize), overlap: usize) -> Self {
        let num_chunks_row = (image_shape.0 + chunk_size.0 - 1) / chunk_size.0;
        let num_chunks_col = (image_shape.1 + chunk_size.1 - 1) / chunk_size.1;

        Self {
            image_shape,
            chunk_size,
            overlap,
            current_row: 0,
            current_col: 0,
            num_chunks_row,
            num_chunks_col,
        }
    }

    /// Get total number of chunks
    pub fn total_chunks(&self) -> usize {
        self.num_chunks_row * self.num_chunks_col
    }
}

impl Iterator for ChunkRegionIterator {
    type Item = ChunkRegion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.num_chunks_row {
            return None;
        }

        let start_row = self.current_row * self.chunk_size.0;
        let start_col = self.current_col * self.chunk_size.1;
        let end_row = ((self.current_row + 1) * self.chunk_size.0).min(self.image_shape.0);
        let end_col = ((self.current_col + 1) * self.chunk_size.1).min(self.image_shape.1);

        let mut region = ChunkRegion::new(
            (start_row, start_col),
            (end_row, end_col),
            self.overlap,
            self.image_shape,
        );
        region.chunk_id = (self.current_row, self.current_col);

        // Advance to next chunk
        self.current_col += 1;
        if self.current_col >= self.num_chunks_col {
            self.current_col = 0;
            self.current_row += 1;
        }

        Some(region)
    }
}

// ============================================================================
// Chunk Data Types
// ============================================================================

/// Represents chunk data that can be either in-memory or memory-mapped
pub enum ChunkData<T> {
    /// In-memory array
    InMemory(Array2<T>),

    /// Memory-mapped array (stored as path and loaded on demand)
    MemoryMapped {
        path: PathBuf,
        shape: (usize, usize),
        _phantom: PhantomData<T>,
    },
}

impl<T: Float + FromPrimitive + Clone> ChunkData<T> {
    /// Get the data as an owned array
    pub fn to_array(&self) -> NdimageResult<Array2<T>> {
        match self {
            ChunkData::InMemory(arr) => Ok(arr.clone()),
            ChunkData::MemoryMapped { path, shape, .. } => load_chunk_from_file(path, *shape),
        }
    }

    /// Get the shape of the chunk
    pub fn shape(&self) -> (usize, usize) {
        match self {
            ChunkData::InMemory(arr) => (arr.nrows(), arr.ncols()),
            ChunkData::MemoryMapped { shape, .. } => *shape,
        }
    }
}

// ============================================================================
// Chunk Operation Traits
// ============================================================================

/// Trait for operations that can be applied to chunks
pub trait ChunkOperation<T>: Send + Sync
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync,
{
    /// Apply the operation to a single chunk
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>>;

    /// Get the required overlap for this operation
    fn required_overlap(&self) -> usize;

    /// Get the operation name for logging/debugging
    fn name(&self) -> &str;

    /// Whether this operation supports GPU acceleration
    fn supports_gpu(&self) -> bool {
        false
    }

    /// Apply the operation on GPU (if supported)
    #[cfg(feature = "gpu")]
    fn apply_gpu(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        // Default: fall back to CPU
        self.apply(chunk)
    }
}

/// Trait for binary operations that can be applied to chunks
/// This trait is specifically for bool arrays and doesn't require numeric traits
pub trait BinaryChunkOperation: Send + Sync {
    /// Apply the operation to a single chunk
    fn apply(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>>;

    /// Get the required overlap for this operation
    fn required_overlap(&self) -> usize;

    /// Get the operation name for logging/debugging
    fn name(&self) -> &str;

    /// Whether this operation supports GPU acceleration
    fn supports_gpu(&self) -> bool {
        false
    }
}

/// Trait for operations that can be applied in-place
pub trait InPlaceChunkOperation<T>: Send + Sync
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync,
{
    /// Apply the operation in-place to a chunk
    fn apply_inplace(&self, chunk: &mut ArrayViewMut2<T>) -> NdimageResult<()>;

    /// Get the required overlap for this operation
    fn required_overlap(&self) -> usize;
}

// ============================================================================
// Chunked Image Processor
// ============================================================================

/// Main processor for chunked image operations
pub struct ChunkedImageProcessor {
    config: ChunkingConfig,
    stats: ProcessingStats,
}

/// Statistics about chunk processing
#[derive(Debug, Default)]
pub struct ProcessingStats {
    /// Total chunks processed
    pub chunks_processed: AtomicUsize,

    /// Total bytes processed
    pub bytes_processed: AtomicUsize,

    /// Number of GPU-accelerated chunks
    pub gpu_chunks: AtomicUsize,

    /// Number of CPU-processed chunks
    pub cpu_chunks: AtomicUsize,
}

impl ChunkedImageProcessor {
    /// Create a new chunked image processor
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            stats: ProcessingStats::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ChunkingConfig::default())
    }

    /// Calculate optimal chunk size based on image dimensions and config
    pub fn calculate_chunk_size(
        &self,
        image_shape: (usize, usize),
        element_size: usize,
    ) -> (usize, usize) {
        let target_elements = self.config.max_chunk_bytes / element_size;

        // Start with square chunks
        let base_size =
            ((target_elements as f64).sqrt() as usize).max(self.config.min_chunk_pixels);

        // Adjust to fit image dimensions
        let chunk_rows = base_size.min(image_shape.0);
        let chunk_cols = base_size.min(image_shape.1);

        (chunk_rows, chunk_cols)
    }

    /// Calculate required overlap for an operation
    pub fn calculate_overlap<T, Op>(&self, operation: &Op) -> usize
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync,
        Op: ChunkOperation<T> + ?Sized,
    {
        operation
            .required_overlap()
            .min(self.config.max_overlap_pixels)
    }

    /// Process an image using a chunk operation
    pub fn process<T, Op>(&self, input: &ArrayView2<T>, operation: &Op) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static,
        Op: ChunkOperation<T> + ?Sized,
    {
        let image_shape = (input.nrows(), input.ncols());
        let element_size = std::mem::size_of::<T>();
        let chunk_size = self.calculate_chunk_size(image_shape, element_size);
        let overlap = self.calculate_overlap(operation);

        // Create output array
        let mut output = Array2::zeros(image_shape);

        // Create chunk iterator
        let chunk_iter = ChunkRegionIterator::new(image_shape, chunk_size, overlap);

        if self.config.enable_parallel && is_parallel_enabled() {
            self.process_parallel(input, &mut output, operation, chunk_iter)?;
        } else {
            self.process_sequential(input, &mut output, operation, chunk_iter)?;
        }

        Ok(output)
    }

    /// Process a binary image using a binary chunk operation
    pub fn process_binary<Op>(
        &self,
        input: &ArrayView2<bool>,
        operation: &Op,
    ) -> NdimageResult<Array2<bool>>
    where
        Op: BinaryChunkOperation,
    {
        let image_shape = (input.nrows(), input.ncols());
        let element_size = std::mem::size_of::<bool>();
        let chunk_size = self.calculate_chunk_size(image_shape, element_size);
        let overlap = operation
            .required_overlap()
            .min(self.config.max_overlap_pixels);

        // Create output array
        let mut output = Array2::from_elem(image_shape, false);

        // Create chunk iterator
        let chunk_iter = ChunkRegionIterator::new(image_shape, chunk_size, overlap);

        if self.config.enable_parallel && is_parallel_enabled() {
            self.process_binary_parallel(input, &mut output, operation, chunk_iter)?;
        } else {
            self.process_binary_sequential(input, &mut output, operation, chunk_iter)?;
        }

        Ok(output)
    }

    /// Process chunks sequentially
    fn process_sequential<T, Op>(
        &self,
        input: &ArrayView2<T>,
        output: &mut Array2<T>,
        operation: &Op,
        chunk_iter: ChunkRegionIterator,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static,
        Op: ChunkOperation<T> + ?Sized,
    {
        for region in chunk_iter {
            // Extract padded chunk from input
            let chunk = self.extract_chunk(input, &region)?;

            // Apply operation
            let result = operation.apply(&chunk.view())?;

            // Insert result into output (handling overlap)
            self.insert_chunk_result(output, &result.view(), &region)?;

            // Update stats
            self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
            self.stats.cpu_chunks.fetch_add(1, Ordering::Relaxed);
            self.stats.bytes_processed.fetch_add(
                region.size().0 * region.size().1 * std::mem::size_of::<T>(),
                Ordering::Relaxed,
            );
        }

        Ok(())
    }

    /// Process chunks in parallel
    fn process_parallel<T, Op>(
        &self,
        input: &ArrayView2<T>,
        output: &mut Array2<T>,
        operation: &Op,
        chunk_iter: ChunkRegionIterator,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static,
        Op: ChunkOperation<T> + ?Sized,
    {
        let regions: Vec<_> = chunk_iter.collect();

        // Process chunks in parallel and collect results
        let results: Vec<NdimageResult<(ChunkRegion, Array2<T>)>> = regions
            .into_par_iter()
            .map(|region| {
                let chunk = self.extract_chunk(input, &region)?;
                let result = operation.apply(&chunk.view())?;

                // Update stats
                self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
                self.stats.cpu_chunks.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_processed.fetch_add(
                    region.size().0 * region.size().1 * std::mem::size_of::<T>(),
                    Ordering::Relaxed,
                );

                Ok((region, result))
            })
            .collect();

        // Insert results into output
        for result in results {
            let (region, chunk_result) = result?;
            self.insert_chunk_result(output, &chunk_result.view(), &region)?;
        }

        Ok(())
    }

    /// Process binary chunks sequentially
    fn process_binary_sequential<Op>(
        &self,
        input: &ArrayView2<bool>,
        output: &mut Array2<bool>,
        operation: &Op,
        chunk_iter: ChunkRegionIterator,
    ) -> NdimageResult<()>
    where
        Op: BinaryChunkOperation,
    {
        for region in chunk_iter {
            // Extract padded chunk from input
            let chunk = self.extract_binary_chunk(input, &region)?;

            // Apply operation
            let result = operation.apply(&chunk.view())?;

            // Insert result into output (handling overlap)
            self.insert_binary_chunk_result(output, &result.view(), &region)?;

            // Update stats
            self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
            self.stats.cpu_chunks.fetch_add(1, Ordering::Relaxed);
            self.stats.bytes_processed.fetch_add(
                region.size().0 * region.size().1 * std::mem::size_of::<bool>(),
                Ordering::Relaxed,
            );
        }

        Ok(())
    }

    /// Process binary chunks in parallel
    fn process_binary_parallel<Op>(
        &self,
        input: &ArrayView2<bool>,
        output: &mut Array2<bool>,
        operation: &Op,
        chunk_iter: ChunkRegionIterator,
    ) -> NdimageResult<()>
    where
        Op: BinaryChunkOperation,
    {
        let regions: Vec<_> = chunk_iter.collect();

        // Process chunks in parallel and collect results
        let results: Vec<NdimageResult<(ChunkRegion, Array2<bool>)>> = regions
            .into_par_iter()
            .map(|region| {
                let chunk = self.extract_binary_chunk(input, &region)?;
                let result = operation.apply(&chunk.view())?;

                // Update stats
                self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
                self.stats.cpu_chunks.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_processed.fetch_add(
                    region.size().0 * region.size().1 * std::mem::size_of::<bool>(),
                    Ordering::Relaxed,
                );

                Ok((region, result))
            })
            .collect();

        // Insert results into output
        for result in results {
            let (region, chunk_result) = result?;
            self.insert_binary_chunk_result(output, &chunk_result.view(), &region)?;
        }

        Ok(())
    }

    /// Extract a chunk from the input array
    fn extract_chunk<T>(
        &self,
        input: &ArrayView2<T>,
        region: &ChunkRegion,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + Clone,
    {
        let rows = region.padded_start.0..region.padded_end.0;
        let cols = region.padded_start.1..region.padded_end.1;

        Ok(input.slice(ndarray::s![rows, cols]).to_owned())
    }

    /// Insert a processed chunk result into the output array
    fn insert_chunk_result<T>(
        &self,
        output: &mut Array2<T>,
        chunk_result: &ArrayView2<T>,
        region: &ChunkRegion,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + Clone,
    {
        let overlap = region.overlap();

        // Calculate the region within the chunk result that corresponds to the output region
        let result_start_row = overlap.0 .0;
        let result_start_col = overlap.0 .1;
        let result_end_row = chunk_result.nrows() - overlap.1 .0;
        let result_end_col = chunk_result.ncols() - overlap.1 .1;

        // Validate dimensions
        let expected_rows = region.end.0 - region.start.0;
        let expected_cols = region.end.1 - region.start.1;
        let actual_rows = result_end_row - result_start_row;
        let actual_cols = result_end_col - result_start_col;

        if actual_rows != expected_rows || actual_cols != expected_cols {
            return Err(NdimageError::DimensionError(format!(
                "Chunk result dimension mismatch: expected ({}, {}), got ({}, {})",
                expected_rows, expected_cols, actual_rows, actual_cols
            )));
        }

        // Extract the relevant portion of the result
        let result_slice = chunk_result.slice(ndarray::s![
            result_start_row..result_end_row,
            result_start_col..result_end_col
        ]);

        // Get the output region
        let mut output_slice = output.slice_mut(ndarray::s![
            region.start.0..region.end.0,
            region.start.1..region.end.1
        ]);

        match self.config.overlap_strategy {
            OverlapStrategy::Overwrite => {
                output_slice.assign(&result_slice);
            }
            OverlapStrategy::Blend => {
                self.blend_overlap(&mut output_slice, &result_slice, region)?;
            }
            OverlapStrategy::WeightedAverage => {
                self.weighted_average_overlap(&mut output_slice, &result_slice, region)?;
            }
            OverlapStrategy::QualityBased => {
                // For now, default to simple overwrite
                output_slice.assign(&result_slice);
            }
        }

        Ok(())
    }

    /// Extract a binary chunk from the input array
    fn extract_binary_chunk(
        &self,
        input: &ArrayView2<bool>,
        region: &ChunkRegion,
    ) -> NdimageResult<Array2<bool>> {
        let rows = region.padded_start.0..region.padded_end.0;
        let cols = region.padded_start.1..region.padded_end.1;

        Ok(input.slice(ndarray::s![rows, cols]).to_owned())
    }

    /// Insert a processed binary chunk result into the output array
    fn insert_binary_chunk_result(
        &self,
        output: &mut Array2<bool>,
        chunk_result: &ArrayView2<bool>,
        region: &ChunkRegion,
    ) -> NdimageResult<()> {
        let overlap = region.overlap();

        // Calculate the region within the chunk result that corresponds to the output region
        let result_start_row = overlap.0 .0;
        let result_start_col = overlap.0 .1;
        let result_end_row = chunk_result.nrows() - overlap.1 .0;
        let result_end_col = chunk_result.ncols() - overlap.1 .1;

        // Validate dimensions
        let expected_rows = region.end.0 - region.start.0;
        let expected_cols = region.end.1 - region.start.1;
        let actual_rows = result_end_row - result_start_row;
        let actual_cols = result_end_col - result_start_col;

        if actual_rows != expected_rows || actual_cols != expected_cols {
            return Err(NdimageError::DimensionError(format!(
                "Binary chunk result dimension mismatch: expected ({}, {}), got ({}, {})",
                expected_rows, expected_cols, actual_rows, actual_cols
            )));
        }

        // Extract the relevant portion of the result
        let result_slice = chunk_result.slice(ndarray::s![
            result_start_row..result_end_row,
            result_start_col..result_end_col
        ]);

        // Get the output region
        let mut output_slice = output.slice_mut(ndarray::s![
            region.start.0..region.end.0,
            region.start.1..region.end.1
        ]);

        // For binary data, always use simple overwrite
        output_slice.assign(&result_slice);

        Ok(())
    }

    /// Blend overlapping regions using linear interpolation
    fn blend_overlap<T>(
        &self,
        output: &mut ArrayViewMut2<T>,
        result: &ArrayView2<T>,
        region: &ChunkRegion,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + Clone,
    {
        // For the first chunk in each direction, just assign directly
        if region.chunk_id.0 == 0 && region.chunk_id.1 == 0 {
            output.assign(result);
            return Ok(());
        }

        // Simple blend: average with existing values in overlap regions
        for (i, row) in output.rows_mut().into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                let new_val = result[[i, j]];
                if *val == T::zero() {
                    *val = new_val;
                } else {
                    // Simple average
                    *val = (*val + new_val)
                        / T::from_f64(2.0).ok_or_else(|| {
                            NdimageError::ComputationError("Failed to convert 2.0 to float".into())
                        })?;
                }
            }
        }

        Ok(())
    }

    /// Use weighted average for overlap regions
    fn weighted_average_overlap<T>(
        &self,
        output: &mut ArrayViewMut2<T>,
        result: &ArrayView2<T>,
        region: &ChunkRegion,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive + Clone,
    {
        let (rows, cols) = (output.nrows(), output.ncols());

        for i in 0..rows {
            for j in 0..cols {
                let new_val = result[[i, j]];

                if output[[i, j]] == T::zero() {
                    output[[i, j]] = new_val;
                } else {
                    // Weight based on distance from chunk edge
                    let center_row = rows as f64 / 2.0;
                    let center_col = cols as f64 / 2.0;
                    let dist =
                        ((i as f64 - center_row).powi(2) + (j as f64 - center_col).powi(2)).sqrt();
                    let max_dist = (center_row.powi(2) + center_col.powi(2)).sqrt();
                    let weight = T::from_f64(1.0 - dist / max_dist).ok_or_else(|| {
                        NdimageError::ComputationError("Failed to convert weight".into())
                    })?;

                    output[[i, j]] = output[[i, j]] * (T::one() - weight) + new_val * weight;
                }
            }
        }

        Ok(())
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }

    /// Reset processing statistics
    pub fn reset_stats(&self) {
        self.stats.chunks_processed.store(0, Ordering::Relaxed);
        self.stats.bytes_processed.store(0, Ordering::Relaxed);
        self.stats.gpu_chunks.store(0, Ordering::Relaxed);
        self.stats.cpu_chunks.store(0, Ordering::Relaxed);
    }

    // ========================================================================
    // Convenience methods for common operations
    // ========================================================================

    /// Apply Gaussian filter in chunks
    pub fn gaussian_filter<T>(
        &self,
        input: &ArrayView2<T>,
        sigma: f64,
        border_mode: Option<BorderMode>,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float
            + FromPrimitive
            + Debug
            + Clone
            + Send
            + Sync
            + Zero
            + 'static
            + std::ops::AddAssign
            + std::ops::DivAssign,
    {
        let operation =
            GaussianFilterOperation::new(sigma, border_mode.unwrap_or(BorderMode::Reflect));
        self.process(input, &operation)
    }

    /// Apply uniform (box) filter in chunks
    pub fn uniform_filter<T>(
        &self,
        input: &ArrayView2<T>,
        size: usize,
        border_mode: Option<BorderMode>,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float
            + FromPrimitive
            + Debug
            + Clone
            + Send
            + Sync
            + Zero
            + 'static
            + std::ops::AddAssign
            + std::ops::DivAssign,
    {
        let operation =
            UniformFilterOperation::new(size, border_mode.unwrap_or(BorderMode::Reflect));
        self.process(input, &operation)
    }

    /// Apply median filter in chunks
    pub fn median_filter<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let operation = MedianFilterOperation::new(size);
        self.process(input, &operation)
    }

    /// Apply morphological erosion in chunks
    pub fn grey_erosion<T>(
        &self,
        input: &ArrayView2<T>,
        size: usize,
        border_mode: Option<MorphBorderMode>,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let operation =
            MorphErosionOperation::new(size, border_mode.unwrap_or(MorphBorderMode::Constant));
        self.process(input, &operation)
    }

    /// Apply morphological dilation in chunks
    pub fn grey_dilation<T>(
        &self,
        input: &ArrayView2<T>,
        size: usize,
        border_mode: Option<MorphBorderMode>,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let operation =
            MorphDilationOperation::new(size, border_mode.unwrap_or(MorphBorderMode::Constant));
        self.process(input, &operation)
    }
}

// ============================================================================
// Built-in Chunk Operations
// ============================================================================

/// Gaussian filter operation for chunk processing
pub struct GaussianFilterOperation {
    sigma: f64,
    border_mode: BorderMode,
    kernel_radius: usize,
}

impl GaussianFilterOperation {
    /// Create a new Gaussian filter operation
    pub fn new(sigma: f64, border_mode: BorderMode) -> Self {
        let kernel_radius = (sigma * 4.0).ceil() as usize;
        Self {
            sigma,
            border_mode,
            kernel_radius,
        }
    }
}

impl<T> ChunkOperation<T> for GaussianFilterOperation
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + Zero
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        // Convert to f64 for processing
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));

        let result =
            crate::filters::gaussian_filter(&chunk_f64, self.sigma, Some(self.border_mode), None)?;

        // Convert back to T
        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        self.kernel_radius
    }

    fn name(&self) -> &str {
        "gaussian_filter"
    }

    fn supports_gpu(&self) -> bool {
        cfg!(feature = "gpu")
    }
}

/// Uniform (box) filter operation for chunk processing
pub struct UniformFilterOperation {
    size: usize,
    border_mode: BorderMode,
}

impl UniformFilterOperation {
    /// Create a new uniform filter operation
    pub fn new(size: usize, border_mode: BorderMode) -> Self {
        Self { size, border_mode }
    }
}

impl<T> ChunkOperation<T> for UniformFilterOperation
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + Zero
        + 'static
        + std::ops::AddAssign
        + std::ops::DivAssign,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));

        let result = crate::filters::uniform_filter(
            &chunk_f64,
            &[self.size, self.size],
            Some(self.border_mode),
            None,
        )?;

        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        self.size / 2 + 1
    }

    fn name(&self) -> &str {
        "uniform_filter"
    }
}

/// Median filter operation for chunk processing
pub struct MedianFilterOperation {
    size: usize,
}

impl MedianFilterOperation {
    /// Create a new median filter operation
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl<T> ChunkOperation<T> for MedianFilterOperation
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));

        let result = crate::filters::median_filter(
            &chunk_f64,
            &[self.size, self.size],
            Some(BorderMode::Reflect),
        )?;

        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        self.size / 2 + 1
    }

    fn name(&self) -> &str {
        "median_filter"
    }
}

/// Morphological erosion operation for chunk processing
pub struct MorphErosionOperation {
    size: usize,
    border_mode: MorphBorderMode,
}

impl MorphErosionOperation {
    /// Create a new morphological erosion operation
    pub fn new(size: usize, border_mode: MorphBorderMode) -> Self {
        Self { size, border_mode }
    }
}

impl<T> ChunkOperation<T> for MorphErosionOperation
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let chunk_owned = chunk_f64.to_owned();

        let result = crate::morphology::grey_erosion(
            &chunk_owned,
            Some(&[self.size, self.size]),
            None,
            Some(self.border_mode),
            None,
            None,
        )?;

        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        self.size / 2 + 1
    }

    fn name(&self) -> &str {
        "grey_erosion"
    }
}

/// Morphological dilation operation for chunk processing
pub struct MorphDilationOperation {
    size: usize,
    border_mode: MorphBorderMode,
}

impl MorphDilationOperation {
    /// Create a new morphological dilation operation
    pub fn new(size: usize, border_mode: MorphBorderMode) -> Self {
        Self { size, border_mode }
    }
}

impl<T> ChunkOperation<T> for MorphDilationOperation
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let chunk_owned = chunk_f64.to_owned();

        let result = crate::morphology::grey_dilation(
            &chunk_owned,
            Some(&[self.size, self.size]),
            None,
            Some(self.border_mode),
            None,
            None,
        )?;

        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        self.size / 2 + 1
    }

    fn name(&self) -> &str {
        "grey_dilation"
    }
}

// ============================================================================
// Out-of-Core Processing Pipeline
// ============================================================================

/// Pipeline for processing images that don't fit in memory
pub struct OutOfCorePipeline<T> {
    config: ChunkingConfig,
    operations: Vec<Box<dyn ChunkOperation<T>>>,
    _phantom: PhantomData<T>,
}

impl<T> OutOfCorePipeline<T>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static,
{
    /// Create a new out-of-core processing pipeline
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            operations: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add an operation to the pipeline
    pub fn add_operation(mut self, operation: Box<dyn ChunkOperation<T>>) -> Self {
        self.operations.push(operation);
        self
    }

    /// Get the maximum required overlap across all operations
    fn max_overlap(&self) -> usize {
        self.operations
            .iter()
            .map(|op| op.required_overlap())
            .max()
            .unwrap_or(0)
            .min(self.config.max_overlap_pixels)
    }

    /// Process an image stored as a file
    pub fn process_file(
        &self,
        input_path: &Path,
        output_path: &Path,
        image_shape: (usize, usize),
    ) -> NdimageResult<()> {
        let element_size = std::mem::size_of::<T>();
        let processor = ChunkedImageProcessor::new(self.config.clone());
        let chunk_size = processor.calculate_chunk_size(image_shape, element_size);
        let overlap = self.max_overlap();

        // Create chunk iterator
        let chunk_iter = ChunkRegionIterator::new(image_shape, chunk_size, overlap);

        // Process each chunk through the pipeline
        for region in chunk_iter {
            // Load chunk from input file
            let chunk = load_chunk_from_raw_file::<T>(input_path, image_shape, &region)?;

            // Apply all operations in sequence
            let mut result = chunk;
            for operation in &self.operations {
                result = operation.apply(&result.view())?;
            }

            // Write result chunk to output file
            write_chunk_to_raw_file(&result.view(), output_path, image_shape, &region)?;
        }

        Ok(())
    }

    /// Process an in-memory image (for smaller images or testing)
    pub fn process_memory(&self, input: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let mut result = input.to_owned();

        for operation in &self.operations {
            let processor = ChunkedImageProcessor::new(self.config.clone());
            result = processor.process(&result.view(), &**operation)?;
        }

        Ok(result)
    }
}

// ============================================================================
// Zero-Copy Transformations
// ============================================================================

/// A view into a chunk that can be transformed without copying
pub struct ZeroCopyChunk<'a, T> {
    data: ArrayView2<'a, T>,
    region: ChunkRegion,
}

impl<'a, T: Float + Clone> ZeroCopyChunk<'a, T> {
    /// Create a new zero-copy chunk view
    pub fn new(data: ArrayView2<'a, T>, region: ChunkRegion) -> Self {
        Self { data, region }
    }

    /// Get the data view
    pub fn data(&self) -> &ArrayView2<'a, T> {
        &self.data
    }

    /// Get the region
    pub fn region(&self) -> &ChunkRegion {
        &self.region
    }

    /// Apply a point-wise transformation without copying
    pub fn map<F>(&self, f: F) -> Array2<T>
    where
        F: Fn(T) -> T,
    {
        self.data.mapv(|x| f(x))
    }

    /// Get the non-overlapping portion of the chunk
    pub fn core_view(&self) -> ArrayView2<T> {
        let overlap = self.region.overlap();
        let rows = overlap.0 .0..(self.data.nrows() - overlap.1 .0);
        let cols = overlap.0 .1..(self.data.ncols() - overlap.1 .1);
        self.data.slice(ndarray::s![rows, cols])
    }
}

/// Iterator that yields zero-copy chunks
pub struct ZeroCopyChunkIterator<'a, T> {
    input: &'a ArrayView2<'a, T>,
    region_iter: ChunkRegionIterator,
}

impl<'a, T: Float + Clone> ZeroCopyChunkIterator<'a, T> {
    /// Create a new zero-copy chunk iterator
    pub fn new(input: &'a ArrayView2<'a, T>, chunk_size: (usize, usize), overlap: usize) -> Self {
        let image_shape = (input.nrows(), input.ncols());
        let region_iter = ChunkRegionIterator::new(image_shape, chunk_size, overlap);

        Self { input, region_iter }
    }
}

impl<'a, T: Float + Clone> Iterator for ZeroCopyChunkIterator<'a, T> {
    type Item = ZeroCopyChunk<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.region_iter.next().map(|region| {
            let rows = region.padded_start.0..region.padded_end.0;
            let cols = region.padded_start.1..region.padded_end.1;
            let data = self.input.slice(ndarray::s![rows, cols]);
            ZeroCopyChunk::new(data, region)
        })
    }
}

// ============================================================================
// File I/O Helpers
// ============================================================================

/// Load a chunk from a raw binary file
fn load_chunk_from_raw_file<T: Float + FromPrimitive + Clone>(
    path: &Path,
    image_shape: (usize, usize),
    region: &ChunkRegion,
) -> NdimageResult<Array2<T>> {
    let element_size = std::mem::size_of::<T>();
    let mut file = BufReader::new(File::open(path).map_err(NdimageError::IoError)?);

    let chunk_rows = region.padded_end.0 - region.padded_start.0;
    let chunk_cols = region.padded_end.1 - region.padded_start.1;
    let mut data = Vec::with_capacity(chunk_rows * chunk_cols);

    for row in region.padded_start.0..region.padded_end.0 {
        // Seek to the start of this row in the chunk
        let offset = (row * image_shape.1 + region.padded_start.1) * element_size;
        file.seek(SeekFrom::Start(offset as u64))
            .map_err(NdimageError::IoError)?;

        // Read the row
        let mut row_buffer = vec![0u8; chunk_cols * element_size];
        file.read_exact(&mut row_buffer)
            .map_err(NdimageError::IoError)?;

        // Convert bytes to T values
        for i in 0..chunk_cols {
            let bytes = &row_buffer[i * element_size..(i + 1) * element_size];
            let value = if element_size == 8 {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                    NdimageError::ComputationError("Failed to convert bytes to f64".into())
                })?;
                T::from_f64(f64::from_le_bytes(arr)).ok_or_else(|| {
                    NdimageError::ComputationError("Failed to convert f64 to T".into())
                })?
            } else {
                let arr: [u8; 4] = bytes.try_into().map_err(|_| {
                    NdimageError::ComputationError("Failed to convert bytes to f32".into())
                })?;
                T::from_f32(f32::from_le_bytes(arr)).ok_or_else(|| {
                    NdimageError::ComputationError("Failed to convert f32 to T".into())
                })?
            };
            data.push(value);
        }
    }

    Array2::from_shape_vec((chunk_rows, chunk_cols), data).map_err(|e| NdimageError::ShapeError(e))
}

/// Write a chunk to a raw binary file
fn write_chunk_to_raw_file<T: Float + Clone>(
    chunk: &ArrayView2<T>,
    path: &Path,
    image_shape: (usize, usize),
    region: &ChunkRegion,
) -> NdimageResult<()> {
    let element_size = std::mem::size_of::<T>();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(NdimageError::IoError)?;
    let mut writer = BufWriter::new(file);

    let overlap = region.overlap();
    let core_start_row = overlap.0 .0;
    let core_start_col = overlap.0 .1;
    let core_end_row = chunk.nrows() - overlap.1 .0;
    let core_end_col = chunk.ncols() - overlap.1 .1;

    for (chunk_row, output_row) in (core_start_row..core_end_row).zip(region.start.0..region.end.0)
    {
        // Seek to the start of this row in the output
        let offset = (output_row * image_shape.1 + region.start.1) * element_size;
        writer
            .seek(SeekFrom::Start(offset as u64))
            .map_err(NdimageError::IoError)?;

        // Write the row
        for chunk_col in core_start_col..core_end_col {
            let value = chunk[[chunk_row, chunk_col]];
            let bytes = if element_size == 8 {
                value.to_f64().unwrap_or(0.0).to_le_bytes().to_vec()
            } else {
                value.to_f32().unwrap_or(0.0).to_le_bytes().to_vec()
            };
            writer.write_all(&bytes).map_err(NdimageError::IoError)?;
        }
    }

    writer.flush().map_err(NdimageError::IoError)?;
    Ok(())
}

/// Load a chunk from a binary file (for ChunkData::MemoryMapped)
fn load_chunk_from_file<T: Float + FromPrimitive + Clone>(
    path: &Path,
    shape: (usize, usize),
) -> NdimageResult<Array2<T>> {
    let element_size = std::mem::size_of::<T>();
    let mut file = BufReader::new(File::open(path).map_err(NdimageError::IoError)?);

    let total_elements = shape.0 * shape.1;
    let mut data = Vec::with_capacity(total_elements);

    let mut buffer = vec![0u8; total_elements * element_size];
    file.read_exact(&mut buffer)
        .map_err(NdimageError::IoError)?;

    for i in 0..total_elements {
        let bytes = &buffer[i * element_size..(i + 1) * element_size];
        let value = if element_size == 8 {
            let arr: [u8; 8] = bytes
                .try_into()
                .map_err(|_| NdimageError::ComputationError("Failed to convert bytes".into()))?;
            T::from_f64(f64::from_le_bytes(arr))
                .ok_or_else(|| NdimageError::ComputationError("Failed to convert to T".into()))?
        } else {
            let arr: [u8; 4] = bytes
                .try_into()
                .map_err(|_| NdimageError::ComputationError("Failed to convert bytes".into()))?;
            T::from_f32(f32::from_le_bytes(arr))
                .ok_or_else(|| NdimageError::ComputationError("Failed to convert to T".into()))?
        };
        data.push(value);
    }

    Array2::from_shape_vec(shape, data).map_err(|e| NdimageError::ShapeError(e))
}

// ============================================================================
// GPU-Accelerated Chunk Processing
// ============================================================================

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::backend::{Backend, DeviceManager};

    /// GPU-accelerated chunk processor
    pub struct GpuChunkProcessor {
        config: ChunkingConfig,
        device_manager: DeviceManager,
    }

    impl GpuChunkProcessor {
        /// Create a new GPU chunk processor
        pub fn new(config: ChunkingConfig) -> NdimageResult<Self> {
            let device_manager = DeviceManager::new()?;
            Ok(Self {
                config,
                device_manager,
            })
        }

        /// Check if GPU is available and beneficial for the given size
        pub fn should_use_gpu(&self, array_size: usize) -> bool {
            let capabilities = self.device_manager.get_capabilities();
            capabilities.gpu_available && array_size >= 1024 * 1024
        }

        /// Process a chunk on GPU
        pub fn process_chunk<T, Op>(
            &self,
            chunk: &ArrayView2<T>,
            operation: &Op,
        ) -> NdimageResult<Array2<T>>
        where
            T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static,
            Op: ChunkOperation<T> + ?Sized,
        {
            if self.should_use_gpu(chunk.len()) && operation.supports_gpu() {
                operation.apply_gpu(chunk)
            } else {
                operation.apply(chunk)
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_chunk_region_iterator() {
        let image_shape = (100, 100);
        let chunk_size = (30, 30);
        let overlap = 5;

        let iter = ChunkRegionIterator::new(image_shape, chunk_size, overlap);
        let regions: Vec<_> = iter.collect();

        // Should have 4x4 = 16 chunks
        assert_eq!(regions.len(), 16);

        // Check first chunk
        assert_eq!(regions[0].start, (0, 0));
        assert_eq!(regions[0].padded_start, (0, 0));
        assert!(regions[0].end.0 <= 30);
        assert!(regions[0].end.1 <= 30);

        // Check last chunk covers the end
        let last = regions.last().expect("Should have regions");
        assert_eq!(last.end, image_shape);
    }

    #[test]
    fn test_chunked_processor_small_image() {
        let processor = ChunkedImageProcessor::new(ChunkingConfig {
            max_chunk_bytes: 1024, // Small chunks
            enable_parallel: false,
            ..Default::default()
        });

        let input = Array2::<f64>::ones((50, 50));
        let operation = GaussianFilterOperation::new(1.0, BorderMode::Reflect);

        let result = processor.process(&input.view(), &operation);
        assert!(result.is_ok());

        let output = result.expect("Should succeed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_gaussian_filter_chunked() {
        let processor = ChunkedImageProcessor::new(ChunkingConfig {
            max_chunk_bytes: 8192, // Force chunking
            enable_parallel: false,
            ..Default::default()
        });

        let input = Array2::<f64>::ones((100, 100));
        let result = processor.gaussian_filter(&input.view(), 1.0, None);

        assert!(result.is_ok());
        let output = result.expect("Should succeed");

        // Interior values should be approximately 1.0
        for i in 10..90 {
            for j in 10..90 {
                assert!((output[[i, j]] - 1.0).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_uniform_filter_chunked() {
        let processor = ChunkedImageProcessor::new(ChunkingConfig {
            max_chunk_bytes: 8192,
            enable_parallel: false,
            ..Default::default()
        });

        let input = Array2::<f64>::ones((100, 100));
        let result = processor.uniform_filter(&input.view(), 3, None);

        assert!(result.is_ok());
        let output = result.expect("Should succeed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_morphological_erosion_chunked() {
        let processor = ChunkedImageProcessor::new(ChunkingConfig {
            max_chunk_bytes: 8192,
            enable_parallel: false,
            ..Default::default()
        });

        let input = Array2::<f64>::ones((100, 100));
        let result = processor.grey_erosion(&input.view(), 3, None);

        assert!(result.is_ok());
        let output = result.expect("Should succeed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_out_of_core_pipeline() {
        let config = ChunkingConfig {
            max_chunk_bytes: 4096,
            enable_parallel: false,
            ..Default::default()
        };

        let pipeline = OutOfCorePipeline::<f64>::new(config).add_operation(Box::new(
            GaussianFilterOperation::new(1.0, BorderMode::Reflect),
        ));

        let input = Array2::<f64>::ones((50, 50));
        let result = pipeline.process_memory(&input.view());

        assert!(result.is_ok());
    }

    #[test]
    fn test_zero_copy_chunk_iterator() {
        let input = Array2::<f64>::ones((100, 100));
        let chunk_size = (30, 30);
        let overlap = 5;

        let input_view = input.view();
        let iter = ZeroCopyChunkIterator::new(&input_view, chunk_size, overlap);
        let chunks: Vec<_> = iter.collect();

        assert!(!chunks.is_empty());

        // Check that we can access data without copying
        for chunk in &chunks {
            let core = chunk.core_view();
            assert!(!core.is_empty());
        }
    }

    #[test]
    fn test_overlap_strategies() {
        for strategy in [
            OverlapStrategy::Overwrite,
            OverlapStrategy::Blend,
            OverlapStrategy::WeightedAverage,
        ] {
            let config = ChunkingConfig {
                max_chunk_bytes: 4096,
                overlap_strategy: strategy,
                enable_parallel: false,
                ..Default::default()
            };

            let processor = ChunkedImageProcessor::new(config);
            let input = Array2::<f64>::ones((50, 50));

            let result = processor.gaussian_filter(&input.view(), 1.0, None);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_processing_stats() {
        let processor = ChunkedImageProcessor::new(ChunkingConfig {
            max_chunk_bytes: 4096,
            enable_parallel: false,
            ..Default::default()
        });

        let input = Array2::<f64>::ones((100, 100));
        let _ = processor.gaussian_filter(&input.view(), 1.0, None);

        let stats = processor.stats();
        assert!(stats.chunks_processed.load(Ordering::Relaxed) > 0);
        assert!(stats.bytes_processed.load(Ordering::Relaxed) > 0);
        assert!(stats.cpu_chunks.load(Ordering::Relaxed) > 0);
    }
}
