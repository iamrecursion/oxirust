//! Large dataset support with streaming iterators and zero-copy processing.
//!
//! This module provides efficient processing capabilities for GB-scale datasets
//! that don't fit in memory, using memory-mapped files and streaming iterators.

use super::memmap::MemoryMappedArray;
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use std::marker::PhantomData;
use std::ops::Range;

/// Streaming chunk iterator for large datasets
///
/// This iterator yields slices from a memory-mapped file without copying data.
/// It's optimized for sequential access patterns with minimal memory footprint.
pub struct StreamingChunkIterator<'a, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    /// The underlying memory-mapped array
    mmap: &'a MemoryMappedArray<A>,

    /// Total number of elements
    total_elements: usize,

    /// Current position in the array
    current_position: usize,

    /// Chunk size in elements
    chunk_size: usize,

    /// Phantom data for type parameter
    phantom: PhantomData<A>,
}

impl<'a, A> StreamingChunkIterator<'a, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    /// Create a new streaming chunk iterator
    pub fn new(mmap: &'a MemoryMappedArray<A>, chunk_size: usize) -> Self {
        let total_elements = mmap.shape.iter().product();

        Self {
            mmap,
            total_elements,
            current_position: 0,
            chunk_size,
            phantom: PhantomData,
        }
    }

    /// Get the total number of chunks
    pub fn num_chunks(&self) -> usize {
        self.total_elements.div_ceil(self.chunk_size)
    }

    /// Get the current chunk index
    pub fn current_chunk(&self) -> usize {
        self.current_position / self.chunk_size
    }

    /// Reset iterator to the beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Get a specific chunk by index without advancing the iterator
    pub fn get_chunk(&self, chunk_index: usize) -> Option<&'a [A]> {
        if chunk_index >= self.num_chunks() {
            return None;
        }

        let start = chunk_index * self.chunk_size;
        let end = ((chunk_index + 1) * self.chunk_size).min(self.total_elements);

        let slice = self.mmap.as_slice();
        Some(&slice[start..end])
    }

    /// Get the byte range for a specific chunk
    pub fn chunk_byte_range(&self, chunk_index: usize) -> Option<Range<usize>> {
        if chunk_index >= self.num_chunks() {
            return None;
        }

        let elem_size = std::mem::size_of::<A>();
        let start = chunk_index * self.chunk_size;
        let end = ((chunk_index + 1) * self.chunk_size).min(self.total_elements);

        Some((start * elem_size)..(end * elem_size))
    }
}

impl<'a, A> Iterator for StreamingChunkIterator<'a, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    type Item = &'a [A];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position >= self.total_elements {
            return None;
        }

        let start = self.current_position;
        let end = (self.current_position + self.chunk_size).min(self.total_elements);

        self.current_position = end;

        let slice = self.mmap.as_slice();
        Some(&slice[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_elements - self.current_position;
        let remaining_chunks = remaining.div_ceil(self.chunk_size);
        (remaining_chunks, Some(remaining_chunks))
    }
}

impl<'a, A> ExactSizeIterator for StreamingChunkIterator<'a, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        let remaining = self.total_elements - self.current_position;
        remaining.div_ceil(self.chunk_size)
    }
}

/// Parallel processing support for streaming iterators
#[cfg(feature = "parallel")]
pub struct ParallelStreamingProcessor<'a, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    /// The underlying iterator
    iterator: StreamingChunkIterator<'a, A>,

    /// Number of parallel workers
    num_workers: usize,
}

#[cfg(feature = "parallel")]
impl<'a, A> ParallelStreamingProcessor<'a, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    /// Create a new parallel streaming processor
    pub fn new(mmap: &'a MemoryMappedArray<A>, chunk_size: usize, num_workers: usize) -> Self {
        Self {
            iterator: StreamingChunkIterator::new(mmap, chunk_size),
            num_workers,
        }
    }

    /// Process all chunks in parallel
    pub fn process<F, R>(&self, f: F) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A]) -> R + Send + Sync,
        R: Send,
    {
        use crate::parallel_ops::*;

        let num_chunks = self.iterator.num_chunks();
        let chunk_indices: Vec<usize> = (0..num_chunks).collect();

        let results: Vec<R> = chunk_indices
            .into_par_iter()
            .filter_map(|idx| self.iterator.get_chunk(idx).map(|chunk| f(chunk)))
            .collect();

        Ok(results)
    }

    /// Process chunks in parallel with error handling
    pub fn try_process<F, R, E>(&self, f: F) -> CoreResult<Vec<R>>
    where
        F: Fn(&[A]) -> Result<R, E> + Send + Sync,
        R: Send,
        E: std::fmt::Display + Send,
    {
        use crate::parallel_ops::*;

        let num_chunks = self.iterator.num_chunks();
        let chunk_indices: Vec<usize> = (0..num_chunks).collect();

        let results: Result<Vec<R>, CoreError> = chunk_indices
            .into_par_iter()
            .map(|idx| {
                self.iterator
                    .get_chunk(idx)
                    .ok_or_else(|| {
                        CoreError::IndexError(
                            ErrorContext::new(format!("Chunk {idx} not found"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })
                    .and_then(|chunk| {
                        f(chunk).map_err(|e| {
                            CoreError::InvalidArgument(
                                ErrorContext::new(format!("Processing error: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })
                    })
            })
            .collect();

        results
    }
}

/// Create a streaming iterator for a memory-mapped array
#[allow(dead_code)]
pub fn create_streaming_iterator<A>(
    mmap: &MemoryMappedArray<A>,
    chunk_size: usize,
) -> StreamingChunkIterator<'_, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    StreamingChunkIterator::new(mmap, chunk_size)
}

/// Create a parallel streaming processor
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn create_parallel_processor<A>(
    mmap: &MemoryMappedArray<A>,
    chunk_size: usize,
    num_workers: usize,
) -> ParallelStreamingProcessor<'_, A>
where
    A: Clone + Copy + Send + Sync + 'static,
{
    ParallelStreamingProcessor::new(mmap, chunk_size, num_workers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_efficient::memmap::{create_temp_mmap, AccessMode};
    use crate::ndarray::Array1;

    #[test]
    fn test_streaming_iterator_creation() {
        // Create test data
        let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());

        // Create temporary memory-mapped array
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        // Create streaming iterator with 100-element chunks
        let iterator = StreamingChunkIterator::new(&mmap, 100);

        assert_eq!(iterator.num_chunks(), 10);
        assert_eq!(iterator.current_chunk(), 0);
    }

    #[test]
    fn test_streaming_iterator_iteration() {
        let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let iterator = StreamingChunkIterator::new(&mmap, 100);

        let chunks: Vec<_> = iterator.collect();

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[9].len(), 100);
    }

    #[test]
    fn test_streaming_iterator_get_chunk() {
        let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let iterator = StreamingChunkIterator::new(&mmap, 100);

        // Get specific chunk
        let chunk = iterator.get_chunk(5).expect("Chunk not found");

        assert_eq!(chunk.len(), 100);
        assert!((chunk[0] - 500.0).abs() < 1e-10);
        assert!((chunk[99] - 599.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_iterator_reset() {
        let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let mut iterator = StreamingChunkIterator::new(&mmap, 100);

        // Advance iterator
        let _ = iterator.next();
        let _ = iterator.next();

        assert_eq!(iterator.current_chunk(), 2);

        // Reset
        iterator.reset();

        assert_eq!(iterator.current_chunk(), 0);
    }

    #[test]
    fn test_streaming_iterator_exact_size() {
        let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let iterator = StreamingChunkIterator::new(&mmap, 100);

        assert_eq!(iterator.len(), 10);

        let mut iter = iterator;
        let _ = iter.next();
        assert_eq!(iter.len(), 9);
    }

    #[test]
    fn test_chunk_byte_range() {
        let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let iterator = StreamingChunkIterator::new(&mmap, 100);

        let range = iterator.chunk_byte_range(0).expect("Range not found");
        let elem_size = std::mem::size_of::<f64>();

        assert_eq!(range, 0..(100 * elem_size));
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_processor() {
        let data: Array1<f64> = Array1::from_vec((0..10000).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let processor = ParallelStreamingProcessor::new(&mmap, 1000, 4);

        // Sum all elements in each chunk
        let chunk_sums = processor
            .process(|chunk| chunk.iter().sum::<f64>())
            .expect("Processing failed");

        assert_eq!(chunk_sums.len(), 10);
    }

    #[test]
    fn test_uneven_chunks() {
        // Test with data that doesn't divide evenly by chunk size
        let data: Array1<f64> = Array1::from_vec((0..1050).map(|i| i as f64).collect());
        let mmap = create_temp_mmap(&data, AccessMode::ReadOnly, 0).expect("Failed to create mmap");

        let iterator = StreamingChunkIterator::new(&mmap, 100);

        let chunks: Vec<_> = iterator.collect();

        assert_eq!(chunks.len(), 11);
        assert_eq!(chunks[10].len(), 50); // Last chunk has 50 elements
    }
}
