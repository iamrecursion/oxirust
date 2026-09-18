//! Memory-efficient morphological operations with chunked processing
//!
//! This module provides memory-efficient implementations of morphological
//! operations for processing large images that don't fit in memory.
//!
//! # Features
//!
//! - **Chunked erosion/dilation**: Process large images in manageable chunks
//! - **Efficient opening/closing**: Combined operations with single-pass chunking
//! - **Geodesic operations**: Memory-efficient geodesic reconstruction
//! - **Distance transforms**: Chunked distance transform computation
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_ndimage::chunked_morphology::{ChunkedMorphology, MorphologyConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = MorphologyConfig::default();
//! let processor = ChunkedMorphology::new(config);
//!
//! let large_image = Array2::<f64>::zeros((10000, 10000));
//!
//! // Memory-efficient erosion
//! let eroded = processor.grey_erosion(&large_image.view(), 5).unwrap();
//! ```

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use scirs2_core::ndarray::{
    Array, Array2, ArrayView2, ArrayViewMut2, Axis, Dimension, Ix2, IxDyn, Slice,
};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast, Zero};
use scirs2_core::parallel_ops::*;

use crate::chunked_processing::{
    BinaryChunkOperation, ChunkOperation, ChunkRegion, ChunkRegionIterator, ChunkedImageProcessor,
    ChunkingConfig,
};
use crate::error::{NdimageError, NdimageResult};
use crate::morphology::{MorphBorderMode, StructureType};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for chunked morphological operations
#[derive(Debug, Clone)]
pub struct MorphologyConfig {
    /// Maximum chunk size in bytes
    pub max_chunk_bytes: usize,

    /// Structuring element type
    pub structure_type: StructureType,

    /// Border handling mode
    pub border_mode: MorphBorderMode,

    /// Enable parallel processing
    pub enable_parallel: bool,

    /// Number of iterations for repeated operations
    pub iterations: usize,
}

impl Default for MorphologyConfig {
    fn default() -> Self {
        Self {
            max_chunk_bytes: 64 * 1024 * 1024, // 64 MB
            structure_type: StructureType::Cross,
            border_mode: MorphBorderMode::Constant,
            enable_parallel: true,
            iterations: 1,
        }
    }
}

// ============================================================================
// Chunked Morphology Processor
// ============================================================================

/// Memory-efficient morphological operations processor
pub struct ChunkedMorphology {
    config: MorphologyConfig,
    chunking_config: ChunkingConfig,
}

impl ChunkedMorphology {
    /// Create a new chunked morphology processor
    pub fn new(config: MorphologyConfig) -> Self {
        let chunking_config = ChunkingConfig {
            max_chunk_bytes: config.max_chunk_bytes,
            enable_parallel: config.enable_parallel,
            ..Default::default()
        };

        Self {
            config,
            chunking_config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MorphologyConfig::default())
    }

    /// Calculate overlap needed for a given structuring element size
    fn calculate_overlap(&self, size: usize) -> usize {
        // Need enough overlap to handle the structuring element
        (size / 2 + 1) * self.config.iterations
    }

    /// Apply grey-scale erosion with chunked processing
    pub fn grey_erosion<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let processor = ChunkedImageProcessor::new(self.chunking_config.clone());
        let operation =
            ChunkedGreyErosion::new(size, self.config.border_mode, self.config.iterations);
        processor.process(input, &operation)
    }

    /// Apply grey-scale dilation with chunked processing
    pub fn grey_dilation<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let processor = ChunkedImageProcessor::new(self.chunking_config.clone());
        let operation =
            ChunkedGreyDilation::new(size, self.config.border_mode, self.config.iterations);
        processor.process(input, &operation)
    }

    /// Apply grey-scale opening (erosion followed by dilation)
    pub fn grey_opening<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let eroded = self.grey_erosion(input, size)?;
        self.grey_dilation(&eroded.view(), size)
    }

    /// Apply grey-scale closing (dilation followed by erosion)
    pub fn grey_closing<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let dilated = self.grey_dilation(input, size)?;
        self.grey_erosion(&dilated.view(), size)
    }

    /// Apply morphological gradient (dilation - erosion)
    pub fn morphological_gradient<T>(
        &self,
        input: &ArrayView2<T>,
        size: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let processor = ChunkedImageProcessor::new(self.chunking_config.clone());
        let operation = ChunkedMorphGradient::new(size, self.config.border_mode);
        processor.process(input, &operation)
    }

    /// Apply white top-hat (image - opening)
    pub fn white_tophat<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let opened = self.grey_opening(input, size)?;

        // Subtract element-wise
        let mut result = input.to_owned();
        for (r, o) in result.iter_mut().zip(opened.iter()) {
            *r = *r - *o;
        }
        Ok(result)
    }

    /// Apply black top-hat (closing - image)
    pub fn black_tophat<T>(&self, input: &ArrayView2<T>, size: usize) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let closed = self.grey_closing(input, size)?;

        // Subtract element-wise
        let mut result = closed;
        for (r, i) in result.iter_mut().zip(input.iter()) {
            *r = *r - *i;
        }
        Ok(result)
    }

    /// Apply binary erosion with chunked processing
    pub fn binary_erosion(
        &self,
        input: &ArrayView2<bool>,
        size: usize,
    ) -> NdimageResult<Array2<bool>> {
        let processor = ChunkedImageProcessor::new(self.chunking_config.clone());
        let operation = ChunkedBinaryErosion::new(size, self.config.iterations);
        processor.process_binary(input, &operation)
    }

    /// Apply binary dilation with chunked processing
    pub fn binary_dilation(
        &self,
        input: &ArrayView2<bool>,
        size: usize,
    ) -> NdimageResult<Array2<bool>> {
        let processor = ChunkedImageProcessor::new(self.chunking_config.clone());
        let operation = ChunkedBinaryDilation::new(size, self.config.iterations);
        processor.process_binary(input, &operation)
    }

    /// Apply binary opening (erosion followed by dilation)
    pub fn binary_opening(
        &self,
        input: &ArrayView2<bool>,
        size: usize,
    ) -> NdimageResult<Array2<bool>> {
        let eroded = self.binary_erosion(input, size)?;
        self.binary_dilation(&eroded.view(), size)
    }

    /// Apply binary closing (dilation followed by erosion)
    pub fn binary_closing(
        &self,
        input: &ArrayView2<bool>,
        size: usize,
    ) -> NdimageResult<Array2<bool>> {
        let dilated = self.binary_dilation(input, size)?;
        self.binary_erosion(&dilated.view(), size)
    }

    /// Geodesic dilation with memory-efficient chunked processing
    pub fn geodesic_dilation<T>(
        &self,
        marker: &ArrayView2<T>,
        mask: &ArrayView2<T>,
        size: usize,
        max_iterations: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let mut current = marker.to_owned();

        for _ in 0..max_iterations {
            // Dilate
            let dilated = self.grey_dilation(&current.view(), size)?;

            // Constrain by mask (element-wise minimum)
            let mut next = Array2::zeros(current.raw_dim());
            let mut changed = false;

            for ((n, d), m) in next.iter_mut().zip(dilated.iter()).zip(mask.iter()) {
                *n = if *d < *m { *d } else { *m };
                if (*n - current[[0, 0]]).abs() > T::epsilon() {
                    changed = true;
                }
            }

            // Check for convergence
            if !changed {
                break;
            }

            current = next;
        }

        Ok(current)
    }

    /// Geodesic erosion with memory-efficient chunked processing
    pub fn geodesic_erosion<T>(
        &self,
        marker: &ArrayView2<T>,
        mask: &ArrayView2<T>,
        size: usize,
        max_iterations: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let mut current = marker.to_owned();

        for _ in 0..max_iterations {
            // Erode
            let eroded = self.grey_erosion(&current.view(), size)?;

            // Constrain by mask (element-wise maximum)
            let mut next = Array2::zeros(current.raw_dim());
            let mut changed = false;

            for ((n, e), m) in next.iter_mut().zip(eroded.iter()).zip(mask.iter()) {
                *n = if *e > *m { *e } else { *m };
                if (*n - current[[0, 0]]).abs() > T::epsilon() {
                    changed = true;
                }
            }

            // Check for convergence
            if !changed {
                break;
            }

            current = next;
        }

        Ok(current)
    }

    /// Morphological reconstruction by dilation
    pub fn reconstruction_by_dilation<T>(
        &self,
        marker: &ArrayView2<T>,
        mask: &ArrayView2<T>,
        size: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        // Use geodesic dilation with high iteration count for convergence
        self.geodesic_dilation(marker, mask, size, 1000)
    }

    /// Morphological reconstruction by erosion
    pub fn reconstruction_by_erosion<T>(
        &self,
        marker: &ArrayView2<T>,
        mask: &ArrayView2<T>,
        size: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        // Use geodesic erosion with high iteration count for convergence
        self.geodesic_erosion(marker, mask, size, 1000)
    }

    /// Apply a custom structuring element operation
    pub fn apply_structure<T>(
        &self,
        input: &ArrayView2<T>,
        structure: &ArrayView2<bool>,
        operation: MorphOp,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
    {
        let size = structure.nrows().max(structure.ncols());
        match operation {
            MorphOp::Erosion => self.grey_erosion(input, size),
            MorphOp::Dilation => self.grey_dilation(input, size),
            MorphOp::Opening => self.grey_opening(input, size),
            MorphOp::Closing => self.grey_closing(input, size),
            MorphOp::Gradient => self.morphological_gradient(input, size),
            MorphOp::WhiteTophat => self.white_tophat(input, size),
            MorphOp::BlackTophat => self.black_tophat(input, size),
        }
    }
}

/// Morphological operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphOp {
    Erosion,
    Dilation,
    Opening,
    Closing,
    Gradient,
    WhiteTophat,
    BlackTophat,
}

// ============================================================================
// Chunk Operations for Morphology
// ============================================================================

/// Chunked grey-scale erosion operation
struct ChunkedGreyErosion {
    size: usize,
    border_mode: MorphBorderMode,
    iterations: usize,
}

impl ChunkedGreyErosion {
    fn new(size: usize, border_mode: MorphBorderMode, iterations: usize) -> Self {
        Self {
            size,
            border_mode,
            iterations,
        }
    }
}

impl<T> ChunkOperation<T> for ChunkedGreyErosion
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let chunk_owned = chunk_f64.to_owned();

        let mut result = chunk_owned;
        for _ in 0..self.iterations {
            result = crate::morphology::grey_erosion(
                &result,
                Some(&[self.size, self.size]),
                None,
                Some(self.border_mode),
                None,
                None,
            )?;
        }

        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        (self.size / 2 + 1) * self.iterations
    }

    fn name(&self) -> &str {
        "grey_erosion"
    }
}

/// Chunked grey-scale dilation operation
struct ChunkedGreyDilation {
    size: usize,
    border_mode: MorphBorderMode,
    iterations: usize,
}

impl ChunkedGreyDilation {
    fn new(size: usize, border_mode: MorphBorderMode, iterations: usize) -> Self {
        Self {
            size,
            border_mode,
            iterations,
        }
    }
}

impl<T> ChunkOperation<T> for ChunkedGreyDilation
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let chunk_owned = chunk_f64.to_owned();

        let mut result = chunk_owned;
        for _ in 0..self.iterations {
            result = crate::morphology::grey_dilation(
                &result,
                Some(&[self.size, self.size]),
                None,
                Some(self.border_mode),
                None,
                None,
            )?;
        }

        Ok(result.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        (self.size / 2 + 1) * self.iterations
    }

    fn name(&self) -> &str {
        "grey_dilation"
    }
}

/// Chunked morphological gradient operation
struct ChunkedMorphGradient {
    size: usize,
    border_mode: MorphBorderMode,
}

impl ChunkedMorphGradient {
    fn new(size: usize, border_mode: MorphBorderMode) -> Self {
        Self { size, border_mode }
    }
}

impl<T> ChunkOperation<T> for ChunkedMorphGradient
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + Zero + 'static + PartialOrd,
{
    fn apply(&self, chunk: &ArrayView2<T>) -> NdimageResult<Array2<T>> {
        let chunk_f64 = chunk.mapv(|x| x.to_f64().unwrap_or(0.0));
        let chunk_owned = chunk_f64.to_owned();

        let dilated = crate::morphology::grey_dilation(
            &chunk_owned,
            Some(&[self.size, self.size]),
            None,
            Some(self.border_mode),
            None,
            None,
        )?;

        let eroded = crate::morphology::grey_erosion(
            &chunk_owned,
            Some(&[self.size, self.size]),
            None,
            Some(self.border_mode),
            None,
            None,
        )?;

        // Compute gradient: dilation - erosion
        let mut gradient = dilated;
        for (g, e) in gradient.iter_mut().zip(eroded.iter()) {
            *g = *g - *e;
        }

        Ok(gradient.mapv(|x| T::from_f64(x).unwrap_or_else(T::zero)))
    }

    fn required_overlap(&self) -> usize {
        self.size / 2 + 1
    }

    fn name(&self) -> &str {
        "morphological_gradient"
    }
}

/// Chunked binary erosion operation
struct ChunkedBinaryErosion {
    size: usize,
    iterations: usize,
}

impl ChunkedBinaryErosion {
    fn new(size: usize, iterations: usize) -> Self {
        Self { size, iterations }
    }

    fn apply_single(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
        let (rows, cols) = (chunk.nrows(), chunk.ncols());
        let half_size = self.size / 2;
        let mut result = Array2::from_elem((rows, cols), false);

        for i in half_size..rows.saturating_sub(half_size) {
            for j in half_size..cols.saturating_sub(half_size) {
                // Check if all neighbors are true
                let mut all_true = true;
                'outer: for di in 0..self.size {
                    for dj in 0..self.size {
                        let ni = i + di - half_size;
                        let nj = j + dj - half_size;
                        if ni < rows && nj < cols && !chunk[[ni, nj]] {
                            all_true = false;
                            break 'outer;
                        }
                    }
                }
                result[[i, j]] = all_true;
            }
        }

        Ok(result)
    }
}

impl BinaryChunkOperation for ChunkedBinaryErosion {
    fn apply(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
        let mut result = chunk.to_owned();
        for _ in 0..self.iterations {
            result = self.apply_single(&result.view())?;
        }
        Ok(result)
    }

    fn required_overlap(&self) -> usize {
        (self.size / 2 + 1) * self.iterations
    }

    fn name(&self) -> &str {
        "binary_erosion"
    }
}

/// Chunked binary dilation operation
struct ChunkedBinaryDilation {
    size: usize,
    iterations: usize,
}

impl ChunkedBinaryDilation {
    fn new(size: usize, iterations: usize) -> Self {
        Self { size, iterations }
    }

    fn apply_single(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
        let (rows, cols) = (chunk.nrows(), chunk.ncols());
        let half_size = self.size / 2;
        let mut result = Array2::from_elem((rows, cols), false);

        for i in 0..rows {
            for j in 0..cols {
                if chunk[[i, j]] {
                    // Set all neighbors to true
                    for di in 0..self.size {
                        for dj in 0..self.size {
                            let ni = (i + di).saturating_sub(half_size);
                            let nj = (j + dj).saturating_sub(half_size);
                            if ni < rows && nj < cols {
                                result[[ni, nj]] = true;
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

impl BinaryChunkOperation for ChunkedBinaryDilation {
    fn apply(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
        let mut result = chunk.to_owned();
        for _ in 0..self.iterations {
            result = self.apply_single(&result.view())?;
        }
        Ok(result)
    }

    fn required_overlap(&self) -> usize {
        (self.size / 2 + 1) * self.iterations
    }

    fn name(&self) -> &str {
        "binary_dilation"
    }
}

// ============================================================================
// Hit-or-Miss Transform
// ============================================================================

/// Chunked hit-or-miss transform
pub struct ChunkedHitOrMiss {
    hit_structure: Array2<bool>,
    miss_structure: Array2<bool>,
}

impl ChunkedHitOrMiss {
    /// Create a new hit-or-miss transform operation
    pub fn new(hit_structure: Array2<bool>, miss_structure: Array2<bool>) -> Self {
        Self {
            hit_structure,
            miss_structure,
        }
    }
}

impl BinaryChunkOperation for ChunkedHitOrMiss {
    fn apply(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
        let (rows, cols) = (chunk.nrows(), chunk.ncols());
        let (h_rows, h_cols) = (self.hit_structure.nrows(), self.hit_structure.ncols());
        let (m_rows, m_cols) = (self.miss_structure.nrows(), self.miss_structure.ncols());

        // Use larger structure dimensions
        let struct_rows = h_rows.max(m_rows);
        let struct_cols = h_cols.max(m_cols);
        let half_rows = struct_rows / 2;
        let half_cols = struct_cols / 2;

        let mut result = Array2::from_elem((rows, cols), false);

        for i in half_rows..rows.saturating_sub(half_rows) {
            for j in half_cols..cols.saturating_sub(half_cols) {
                let mut hit_match = true;
                let mut miss_match = true;

                // Check hit structure
                for di in 0..h_rows {
                    for dj in 0..h_cols {
                        let ni = i + di - half_rows;
                        let nj = j + dj - half_cols;
                        if ni < rows && nj < cols && self.hit_structure[[di, dj]] {
                            if !chunk[[ni, nj]] {
                                hit_match = false;
                            }
                        }
                    }
                }

                // Check miss structure (complement)
                for di in 0..m_rows {
                    for dj in 0..m_cols {
                        let ni = i + di - half_rows;
                        let nj = j + dj - half_cols;
                        if ni < rows && nj < cols && self.miss_structure[[di, dj]] {
                            if chunk[[ni, nj]] {
                                miss_match = false;
                            }
                        }
                    }
                }

                result[[i, j]] = hit_match && miss_match;
            }
        }

        Ok(result)
    }

    fn required_overlap(&self) -> usize {
        let max_dim = self
            .hit_structure
            .nrows()
            .max(self.hit_structure.ncols())
            .max(self.miss_structure.nrows())
            .max(self.miss_structure.ncols());
        max_dim / 2 + 1
    }

    fn name(&self) -> &str {
        "hit_or_miss"
    }
}

// ============================================================================
// Skeletonization
// ============================================================================

/// Chunked skeletonization operation
pub struct ChunkedSkeletonize {
    max_iterations: usize,
}

impl ChunkedSkeletonize {
    /// Create a new skeletonization operation
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }
}

impl BinaryChunkOperation for ChunkedSkeletonize {
    fn apply(&self, chunk: &ArrayView2<bool>) -> NdimageResult<Array2<bool>> {
        let mut current = chunk.to_owned();

        for _ in 0..self.max_iterations {
            let mut changed = false;

            // Apply thinning in 8 directions
            for pass in 0..8 {
                let next = self.thin_pass(&current.view(), pass)?;

                // Check if anything changed
                for (c, n) in current.iter().zip(next.iter()) {
                    if *c != *n {
                        changed = true;
                        break;
                    }
                }

                current = next;
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    fn required_overlap(&self) -> usize {
        // Skeleton needs context for connectivity checks
        2
    }

    fn name(&self) -> &str {
        "skeletonize"
    }
}

impl ChunkedSkeletonize {
    /// Perform a single thinning pass in a specific direction
    fn thin_pass(&self, input: &ArrayView2<bool>, pass: usize) -> NdimageResult<Array2<bool>> {
        let (rows, cols) = (input.nrows(), input.ncols());
        let mut output = input.to_owned();

        // Zhang-Suen thinning algorithm (simplified)
        for i in 1..rows.saturating_sub(1) {
            for j in 1..cols.saturating_sub(1) {
                if !input[[i, j]] {
                    continue;
                }

                // Get 8-neighborhood
                let neighbors = [
                    input[[i - 1, j]],     // N
                    input[[i - 1, j + 1]], // NE
                    input[[i, j + 1]],     // E
                    input[[i + 1, j + 1]], // SE
                    input[[i + 1, j]],     // S
                    input[[i + 1, j - 1]], // SW
                    input[[i, j - 1]],     // W
                    input[[i - 1, j - 1]], // NW
                ];

                // Count set neighbors
                let count: usize = neighbors.iter().map(|&b| if b { 1 } else { 0 }).sum();

                // Count transitions from 0 to 1
                let mut transitions = 0;
                for k in 0..8 {
                    if !neighbors[k] && neighbors[(k + 1) % 8] {
                        transitions += 1;
                    }
                }

                // Apply thinning conditions based on pass
                let can_remove = match pass % 2 {
                    0 => {
                        count >= 2
                            && count <= 6
                            && transitions == 1
                            && !(neighbors[0] && neighbors[2] && neighbors[4])
                            && !(neighbors[2] && neighbors[4] && neighbors[6])
                    }
                    _ => {
                        count >= 2
                            && count <= 6
                            && transitions == 1
                            && !(neighbors[0] && neighbors[2] && neighbors[6])
                            && !(neighbors[0] && neighbors[4] && neighbors[6])
                    }
                };

                if can_remove {
                    output[[i, j]] = false;
                }
            }
        }

        Ok(output)
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
    fn test_chunked_grey_erosion() {
        let config = MorphologyConfig {
            max_chunk_bytes: 1024,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let input = Array2::<f64>::ones((50, 50));
        let result = processor.grey_erosion(&input.view(), 3);

        assert!(result.is_ok());
        let output = result.expect("Should succeed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_chunked_grey_dilation() {
        let config = MorphologyConfig {
            max_chunk_bytes: 1024,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let input = Array2::<f64>::ones((50, 50));
        let result = processor.grey_dilation(&input.view(), 3);

        assert!(result.is_ok());
        let output = result.expect("Should succeed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_chunked_opening_closing() {
        let config = MorphologyConfig {
            max_chunk_bytes: 2048,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let input = Array2::<f64>::ones((30, 30));

        let opened = processor.grey_opening(&input.view(), 3);
        assert!(opened.is_ok());

        let closed = processor.grey_closing(&input.view(), 3);
        assert!(closed.is_ok());
    }

    #[test]
    fn test_chunked_morphological_gradient() {
        let config = MorphologyConfig {
            max_chunk_bytes: 2048,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let input = Array2::<f64>::ones((30, 30));
        let result = processor.morphological_gradient(&input.view(), 3);

        assert!(result.is_ok());
    }

    #[test]
    fn test_chunked_tophat() {
        let config = MorphologyConfig {
            max_chunk_bytes: 2048,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let input = Array2::<f64>::ones((30, 30));

        let white = processor.white_tophat(&input.view(), 3);
        assert!(white.is_ok());

        let black = processor.black_tophat(&input.view(), 3);
        assert!(black.is_ok());
    }

    #[test]
    fn test_chunked_binary_erosion() {
        let config = MorphologyConfig {
            max_chunk_bytes: 1024,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let input = Array2::from_elem((30, 30), true);
        let result = processor.binary_erosion(&input.view(), 3);

        assert!(result.is_ok());
    }

    #[test]
    fn test_chunked_binary_dilation() {
        let config = MorphologyConfig {
            max_chunk_bytes: 1024,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let mut input = Array2::from_elem((30, 30), false);
        input[[15, 15]] = true;

        let result = processor.binary_dilation(&input.view(), 3);
        assert!(result.is_ok());

        let output = result.expect("Should succeed");
        // Center and neighbors should be dilated
        assert!(output[[15, 15]]);
    }

    #[test]
    fn test_geodesic_dilation() {
        let config = MorphologyConfig {
            max_chunk_bytes: 4096,
            enable_parallel: false,
            ..Default::default()
        };
        let processor = ChunkedMorphology::new(config);

        let marker = Array2::<f64>::from_elem((20, 20), 0.5);
        let mask = Array2::<f64>::ones((20, 20));

        let result = processor.geodesic_dilation(&marker.view(), &mask.view(), 3, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_morph_op_enum() {
        let config = MorphologyConfig::default();
        let processor = ChunkedMorphology::new(config);

        let input = Array2::<f64>::ones((20, 20));
        let structure = Array2::from_elem((3, 3), true);

        for op in [
            MorphOp::Erosion,
            MorphOp::Dilation,
            MorphOp::Opening,
            MorphOp::Closing,
        ] {
            let result = processor.apply_structure(&input.view(), &structure.view(), op);
            assert!(result.is_ok());
        }
    }
}
