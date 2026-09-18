use super::chunk_format::{detect_format_version, CompressionType, FormatVersion};
use super::chunked::{ChunkedArray, ChunkingStrategy, OPTIMAL_CHUNK_SIZE};
use super::out_of_core_v2::OutOfCoreArrayV2;
use super::validation;
use crate::error::{CoreError, ErrorContext, ErrorLocation};
use ::ndarray::{Array, ArrayBase, Data, Dimension};
use ::serde::{Deserialize, Serialize};
use oxicode::{config, serde as oxicode_serde};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Internal implementation storage for out-of-core arrays
#[derive(Debug)]
enum OutOfCoreImpl<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// V1 format: Full serialization (legacy)
    V1 {
        shape: Vec<usize>,
        file_path: PathBuf,
        strategy: ChunkingStrategy,
        size: usize,
        is_temp: bool,
    },
    /// V2 format: Chunked serialization with efficient seeking
    V2(OutOfCoreArrayV2<A, D>),
}

/// An array that stores data on disk to reduce memory usage
#[derive(Debug)]
pub struct OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Internal implementation (V1 or V2)
    implementation: OutOfCoreImpl<A, D>,
    /// Phantom data for type parameters
    phantom: PhantomData<(A, D)>,
}

impl<A, D> OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        match &self.implementation {
            OutOfCoreImpl::V1 { shape, .. } => shape,
            OutOfCoreImpl::V2(v2) => &v2.header.shape,
        }
    }

    /// Get the file path
    pub fn file_path(&self) -> &Path {
        match &self.implementation {
            OutOfCoreImpl::V1 { file_path, .. } => file_path,
            OutOfCoreImpl::V2(v2) => &v2.file_path,
        }
    }

    /// Get the chunking strategy
    pub fn strategy(&self) -> ChunkingStrategy {
        match &self.implementation {
            OutOfCoreImpl::V1 { strategy, .. } => *strategy,
            OutOfCoreImpl::V2(_) => ChunkingStrategy::Auto, // V2 handles its own chunking
        }
    }

    /// Get the total size in elements
    pub fn size(&self) -> usize {
        match &self.implementation {
            OutOfCoreImpl::V1 { size, .. } => *size,
            OutOfCoreImpl::V2(v2) => v2.header.total_elements,
        }
    }
}

impl<A, D> OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new out-of-core array with the given data and file path
    ///
    /// This uses the V2 format by default for efficient chunk-based access.
    /// Use `new_v1()` if you need backward compatibility with the V1 format.
    pub fn new<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        validation::check_not_empty(data)?;

        // Use V2 format by default with no compression
        let v2 = OutOfCoreArrayV2::new_streaming(data, file_path, strategy, CompressionType::None)?;

        Ok(Self {
            implementation: OutOfCoreImpl::V2(v2),
            phantom: PhantomData,
        })
    }

    /// Create a new out-of-core array using V1 format (legacy)
    ///
    /// This method is provided for backward compatibility. New code should use `new()` instead.
    pub fn new_v1<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        validation::check_not_empty(data)?;

        let shape = data.shape().to_vec();
        let size = data.len();

        // Create file and write data
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

        // Serialize data to file (in chunks if data is large)
        let _chunked = ChunkedArray::new(data.to_owned(), strategy);

        // Note: This is a simplified implementation that writes the entire array at once.
        // A real implementation would write chunks to save memory.
        let cfg = config::standard();
        let serialized = oxicode_serde::encode_to_vec(&data.to_owned(), cfg).map_err(|e| {
            CoreError::ValidationError(
                ErrorContext::new(format!("{e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        file.write_all(&serialized)
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

        Ok(Self {
            implementation: OutOfCoreImpl::V1 {
                shape,
                file_path: file_path.to_path_buf(),
                strategy,
                size,
                is_temp: false,
            },
            phantom: PhantomData,
        })
    }

    /// Open an existing out-of-core array file with automatic version detection
    ///
    /// This method detects whether the file is in V1 or V2 format and loads it accordingly.
    pub fn open(file_path: &Path) -> Result<Self, CoreError> {
        let file = File::open(file_path)
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

        let mut reader = BufReader::new(file);

        // Detect format version
        let version = detect_format_version(&mut reader)?;

        match version {
            FormatVersion::V2 => {
                // Use V2 implementation
                let v2 = OutOfCoreArrayV2::open(file_path)?;
                Ok(Self {
                    implementation: OutOfCoreImpl::V2(v2),
                    phantom: PhantomData,
                })
            }
            FormatVersion::V1 => {
                // Load V1 format - we need to read the entire array to get metadata
                drop(reader);
                let mut file = File::open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                let cfg = config::standard();
                let (array, _len): (Array<A, D>, usize) =
                    oxicode_serde::decode_owned_from_slice(&buffer, cfg).map_err(|e| {
                        CoreError::ValidationError(
                            ErrorContext::new(format!("{e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                let shape = array.shape().to_vec();
                let size = array.len();

                Ok(Self {
                    implementation: OutOfCoreImpl::V1 {
                        shape,
                        file_path: file_path.to_path_buf(),
                        strategy: ChunkingStrategy::Auto,
                        size,
                        is_temp: false,
                    },
                    phantom: PhantomData,
                })
            }
        }
    }

    /// Migrate a V1 format file to V2 format
    ///
    /// This creates a new V2 file at the specified path with the data from this array.
    /// Returns a new `OutOfCoreArray` using the V2 format.
    pub fn migrate_to_v2(
        &self,
        new_file_path: &Path,
        compression: CompressionType,
    ) -> Result<Self, CoreError> {
        // Load the entire array
        let data = self.load()?;

        // Create a new V2 array with the data
        let v2 = OutOfCoreArrayV2::new_streaming(
            &data,
            new_file_path,
            ChunkingStrategy::Auto,
            compression,
        )?;

        Ok(Self {
            implementation: OutOfCoreImpl::V2(v2),
            phantom: PhantomData,
        })
    }

    /// Create a new out-of-core array with a temporary file
    ///
    /// Uses V2 format by default for efficient chunk-based access.
    pub fn new_temp<S>(
        data: &ArrayBase<S, D>,
        strategy: ChunkingStrategy,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let temp_file = NamedTempFile::new()
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;
        let file_path = temp_file.path().to_path_buf();

        // Manually persist the temp file so it stays around after we return
        let _file = temp_file
            .persist(&file_path)
            .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

        // Create V2 array
        let mut v2 =
            OutOfCoreArrayV2::new_streaming(data, &file_path, strategy, CompressionType::None)?;

        // Mark as temporary
        v2.set_temp(true);

        Ok(Self {
            implementation: OutOfCoreImpl::V2(v2),
            phantom: PhantomData,
        })
    }

    /// Load the entire array into memory
    pub fn load(&self) -> Result<Array<A, D>, CoreError> {
        match &self.implementation {
            OutOfCoreImpl::V2(v2) => {
                // Use efficient V2 loading
                v2.load()
            }
            OutOfCoreImpl::V1 { file_path, .. } => {
                // V1 format: load entire file
                let mut file = File::open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                let cfg = config::standard();
                let (array, _len): (Array<A, D>, usize) =
                    oxicode_serde::decode_owned_from_slice(&buffer, cfg).map_err(|e| {
                        CoreError::ValidationError(
                            ErrorContext::new(format!("{e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                Ok(array)
            }
        }
    }

    /// Load a chunk of the array into memory
    ///
    /// For V2 format, this efficiently seeks to and loads only the requested chunk.
    /// For V1 format (legacy), this loads the entire array and extracts the chunk.
    pub fn load_chunk(&self, chunkindex: usize) -> Result<Array<A, D>, CoreError> {
        if chunkindex >= self.num_chunks() {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!(
                    "Chunk index out of bounds: {} >= {}",
                    chunkindex,
                    self.num_chunks()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        match &self.implementation {
            OutOfCoreImpl::V2(v2) => {
                // Use efficient V2 chunk loading - only reads the requested chunk!
                let chunk_data = v2.load_chunk_v2(chunkindex)?;

                // Convert Vec<A> to Array<A, D>
                // We need to determine the chunk shape
                let chunk_shape = self.calculate_chunk_shape(chunkindex, chunk_data.len())?;

                // Create array from data
                let dyn_array =
                    Array::from_shape_vec(crate::ndarray::IxDyn(&chunk_shape), chunk_data)
                        .map_err(|e| {
                            CoreError::DimensionError(
                                ErrorContext::new(format!("Failed to create chunk array: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })?;

                // Convert to target dimension type
                dyn_array.into_dimensionality::<D>().map_err(|e| {
                    CoreError::DimensionError(
                        ErrorContext::new(format!("Failed to convert chunk dimension: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })
            }
            OutOfCoreImpl::V1 {
                file_path,
                strategy,
                size,
                ..
            } => {
                // V1 format: inefficient - must load entire array
                // This is the old behavior kept for backward compatibility

                // Calculate chunk size and offsets
                let chunk_size = self.get_chunk_size();
                let total_size = *size;

                // Calculate start and end indices for this chunk
                let start_idx = chunkindex * chunk_size;
                let end_idx = std::cmp::min((chunkindex + 1) * chunk_size, total_size);
                let actual_chunk_size = end_idx - start_idx;

                // Open the file
                let mut file = File::open(file_path)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Read entire file
                let mut header_buf = Vec::new();
                file.read_to_end(&mut header_buf)
                    .map_err(|e| CoreError::IoError(ErrorContext::new(e.to_string())))?;

                // Deserialize the whole array
                let cfg = config::standard();
                let (fullarray, _len): (Array<A, D>, usize) =
                    oxicode_serde::decode_owned_from_slice(&header_buf, cfg).map_err(|e| {
                        CoreError::ValidationError(
                            ErrorContext::new(format!("{e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                // Extract chunk from full array
                let shape = self.shape();
                let mut chunkshape = shape.to_vec();

                // Adjust the first dimension for this chunk
                if !chunkshape.is_empty() {
                    let first_dim_size =
                        total_size / chunkshape.iter().skip(1).product::<usize>().max(1);
                    let first_dim_chunk_size =
                        actual_chunk_size / chunkshape.iter().skip(1).product::<usize>().max(1);
                    chunkshape[0] = first_dim_chunk_size.min(first_dim_size);
                }

                // Extract chunk data
                let cloned_array = fullarray.clone();
                let chunk_dynamic = cloned_array.to_shape(chunkshape).map_err(|e| {
                    CoreError::DimensionError(
                        ErrorContext::new(format!("{e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

                // Convert to target dimension type
                Self::safe_dimensionality_conversion(chunk_dynamic.to_owned(), "chunk").map_err(
                    |e| {
                        CoreError::DimensionError(
                            ErrorContext::new(format!("{e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    },
                )
            }
        }
    }

    /// Calculate the shape of a chunk given its index and element count
    fn calculate_chunk_shape(
        &self,
        _chunk_index: usize,
        chunk_elements: usize,
    ) -> Result<Vec<usize>, CoreError> {
        let shape = self.shape();
        let mut chunk_shape = shape.to_vec();

        if chunk_shape.is_empty() {
            return Ok(vec![chunk_elements]);
        }

        // Adjust first dimension based on chunk size
        let other_dims_product: usize = chunk_shape.iter().skip(1).product::<usize>().max(1);
        chunk_shape[0] = chunk_elements / other_dims_product;

        Ok(chunk_shape)
    }

    /// Safely convert an array to the target dimension type with detailed error reporting.
    fn safe_dimensionality_conversion(
        array: Array<A, crate::ndarray::IxDyn>,
        context: &str,
    ) -> Result<Array<A, D>, CoreError> {
        let sourceshape = array.shape().to_vec();
        let source_ndim = sourceshape.len();
        let target_ndim = D::NDIM;

        // Handle dynamic dimension target first (IxDyn)
        if target_ndim.is_none() {
            return array.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(
                    ErrorContext::new(format!(
                        "Failed to convert {context} array to dynamic dimension type. Source shape: {sourceshape:?}"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            });
        }

        let target_ndim = target_ndim.expect("Operation failed");

        // Try direct conversion first for exact matches
        if source_ndim == target_ndim {
            return array.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(
                    ErrorContext::new(format!(
                        "Dimension conversion failed for {} array despite matching dimensions ({} -> {}). \
                         Source shape: {:?}, target dimension type: {}",
                        context, source_ndim, target_ndim, sourceshape, std::any::type_name::<D>()
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            });
        }

        // Handle dimension mismatches with simple strategies
        match source_ndim.cmp(&target_ndim) {
            std::cmp::Ordering::Less => {
                // Fewer dimensions than target - try to expand
                Self::try_expand_dimensions(array, context, source_ndim, target_ndim)
            }
            std::cmp::Ordering::Greater => {
                // More dimensions than target - try to squeeze
                Self::try_squeeze_dimensions(array, context, source_ndim, target_ndim)
            }
            std::cmp::Ordering::Equal => {
                // This case is already handled above, but for completeness
                array.into_dimensionality::<D>().map_err(|_| {
                    CoreError::DimensionError(
                        ErrorContext::new(format!(
                            "Unexpected dimension conversion failure for {context} array with matching dimensions. \
                             Source shape: {sourceshape:?}"
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })
            }
        }
    }

    /// Try to squeeze singleton dimensions.
    fn try_squeeze_dimensions(
        array: Array<A, crate::ndarray::IxDyn>,
        context: &str,
        source_dims: usize,
        target_dims: usize,
    ) -> Result<Array<A, D>, CoreError> {
        let sourceshape = array.shape().to_vec();

        // Find and remove singleton dimensions
        let mut squeezedshape = Vec::new();
        let mut removed_dims = 0;
        let dims_to_remove = source_dims - target_dims;

        for &dim_size in &sourceshape {
            if dim_size == 1 && removed_dims < dims_to_remove {
                // Skip singleton dimension
                removed_dims += 1;
            } else {
                squeezedshape.push(dim_size);
            }
        }

        if squeezedshape.len() != target_dims {
            return Err(CoreError::DimensionError(
                ErrorContext::new(format!(
                    "Cannot squeeze {context} array from {source_dims} to {target_dims} dimensions. Source shape: {sourceshape:?}, only {} singleton dimensions available",
                    sourceshape.iter().filter(|&&x| x == 1).count()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Reshape to squeezed shape and convert
        array
            .into_shape_with_order(crate::ndarray::IxDyn(&squeezedshape))
            .map_err(|_| {
                CoreError::DimensionError(
                    ErrorContext::new(format!(
                        "Cannot reshape {context} array from shape {sourceshape:?} to squeezed shape {squeezedshape:?}"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?
            .into_dimensionality::<D>()
            .map_err(|_| {
                CoreError::DimensionError(
                    ErrorContext::new(format!(
                        "Cannot convert squeezed {context} array from {source_dims} to {target_dims} dimensions"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            })
    }

    /// Try to expand dimensions by adding singleton dimensions.
    fn try_expand_dimensions(
        array: Array<A, crate::ndarray::IxDyn>,
        context: &str,
        source_dims: usize,
        target_dims: usize,
    ) -> Result<Array<A, D>, CoreError> {
        let sourceshape = array.shape().to_vec();
        let dims_to_add = target_dims - source_dims;

        if dims_to_add == 0 {
            return array.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(
                    ErrorContext::new(format!(
                        "Failed to convert {context} array despite equal dimensions"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            });
        }

        // Create expanded shape by adding singleton dimensions at the end
        let mut expandedshape = sourceshape.clone();
        expandedshape.extend(std::iter::repeat_n(1, dims_to_add));

        // Try to reshape to expanded shape
        match array
            .clone()
            .into_shape_with_order(crate::ndarray::IxDyn(&expandedshape))
        {
            Ok(reshaped) => reshaped.into_dimensionality::<D>().map_err(|_| {
                CoreError::DimensionError(
                    ErrorContext::new(format!(
                        "Failed to convert expanded {context} array to target dimension type"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            }),
            Err(_) => {
                // Try adding singleton dimensions at the beginning instead
                let mut altshape = vec![1; dims_to_add];
                altshape.extend_from_slice(&sourceshape);

                array
                    .into_shape_with_order(crate::ndarray::IxDyn(&altshape))
                    .map_err(|_| {
                        CoreError::DimensionError(
                            ErrorContext::new(format!(
                                "Cannot reshape {context} array from shape {sourceshape:?} to any expanded shape"
                            ))
                            .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?
                    .into_dimensionality::<D>()
                    .map_err(|_| {
                        CoreError::DimensionError(
                            ErrorContext::new(format!(
                                "Cannot expand {context} array from {source_dims} to {target_dims} dimensions"
                            ))
                            .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })
            }
        }
    }

    /// Get the chunk size based on the chunking strategy
    fn get_chunk_size(&self) -> usize {
        match &self.implementation {
            OutOfCoreImpl::V2(v2) => {
                // V2 manages its own chunks
                if v2.chunk_index.is_empty() {
                    0
                } else {
                    v2.chunk_index
                        .get_entry(0)
                        .map(|entry| entry.num_elements)
                        .unwrap_or(0)
                }
            }
            OutOfCoreImpl::V1 { strategy, size, .. } => {
                let total_size = *size;
                match strategy {
                    ChunkingStrategy::Auto => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),
                    ChunkingStrategy::Fixed(chunk_size) => *chunk_size,
                    ChunkingStrategy::FixedBytes(bytes) => bytes / std::mem::size_of::<A>(),
                    ChunkingStrategy::NumChunks(n) => total_size.div_ceil(*n),
                    ChunkingStrategy::Advanced(_) => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),
                }
            }
        }
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        match &self.implementation {
            OutOfCoreImpl::V2(v2) => v2.num_chunks(),
            OutOfCoreImpl::V1 { size, .. } => {
                let chunk_size = self.get_chunk_size();
                if chunk_size == 0 {
                    0
                } else {
                    size.div_ceil(chunk_size)
                }
            }
        }
    }

    /// Check if the array is temporary
    pub fn is_temp(&self) -> bool {
        match &self.implementation {
            OutOfCoreImpl::V2(v2) => v2.is_temp(),
            OutOfCoreImpl::V1 { is_temp, .. } => *is_temp,
        }
    }

    /// Apply a function to each chunk of the array
    pub fn map<F, B, R>(&self, mut f: F) -> Result<R, CoreError>
    where
        F: FnMut(Array<A, D>) -> B,
        R: FromIterator<B>,
    {
        // Get the total number of chunks
        let num_chunks = self.num_chunks();

        if num_chunks == 0 {
            return Err(CoreError::ValueError(
                ErrorContext::new("Cannot map over an empty array".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Process each chunk sequentially
        let mut results = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            // Load the current chunk
            let chunk = self.load_chunk(chunk_idx)?;

            // Apply the function to the chunk and collect the result
            let result = f(chunk);
            results.push(result);
        }

        // Convert the results into the requested collection type
        Ok(R::from_iter(results))
    }

    /// Apply a function to each chunk of the array in parallel
    #[cfg(feature = "parallel")]
    pub fn par_map<F, B, R>(&self, f: F) -> Result<R, CoreError>
    where
        F: Fn(Array<A, D>) -> B + Send + Sync,
        B: Send,
        R: FromIterator<B> + Send,
        A: Send + Sync,
    {
        use crate::parallel_ops::*;

        // Get the total number of chunks
        let num_chunks = self.num_chunks();

        if num_chunks == 0 {
            return Err(CoreError::ValueError(
                ErrorContext::new("Cannot map over an empty array".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Create an iterator of chunk indices
        let chunk_indices: Vec<usize> = (0..num_chunks).collect();

        // Process each chunk in parallel
        let results: Result<Vec<B>, CoreError> = chunk_indices
            .into_par_iter()
            .map(|chunk_idx| {
                // Load the current chunk
                let chunk = self.load_chunk(chunk_idx)?;

                // Apply the function to the chunk and collect the result
                Ok(f(chunk))
            })
            .collect();

        // Combine all results
        results.map(|vec| vec.into_iter().collect())
    }
}

impl<A, D> Drop for OutOfCoreArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    fn drop(&mut self) {
        // Check if temporary using the accessor method
        let is_temp = self.is_temp();
        let file_path = self.file_path().to_path_buf();

        if is_temp {
            // Attempt to remove the temporary file
            let _ = std::fs::remove_file(&file_path);
        }
    }
}

/// A specialized out-of-core array that uses memory mapping for efficient access
#[derive(Debug)]
pub struct DiskBackedArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// The underlying out-of-core array
    pub array: OutOfCoreArray<A, D>,
    /// Whether the array is read-only
    pub read_only: bool,
}

impl<A, D> DiskBackedArray<A, D>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    /// Create a new disk-backed array with the given data and file path
    pub fn new<S>(
        data: &ArrayBase<S, D>,
        file_path: &Path,
        strategy: ChunkingStrategy,
        read_only: bool,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let array = OutOfCoreArray::new(data, file_path, strategy)?;

        Ok(Self { array, read_only })
    }

    /// Create a new disk-backed array with a temporary file
    pub fn new_temp<S>(
        data: &ArrayBase<S, D>,
        strategy: ChunkingStrategy,
        read_only: bool,
    ) -> Result<Self, CoreError>
    where
        S: Data<Elem = A>,
    {
        let array = OutOfCoreArray::new_temp(data, strategy)?;

        Ok(Self { array, read_only })
    }

    /// Load the entire array into memory
    pub fn load(&self) -> Result<Array<A, D>, CoreError> {
        self.array.load()
    }

    /// Check if the array is temporary
    pub fn is_temp(&self) -> bool {
        self.array.is_temp()
    }
}

/// Create a disk-backed array from the given data and file path
#[allow(dead_code)]
pub fn create_disk_array<A, S, D>(
    data: &ArrayBase<S, D>,
    file_path: &Path,
    strategy: ChunkingStrategy,
    read_only: bool,
) -> Result<DiskBackedArray<A, D>, CoreError>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    S: Data<Elem = A>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    DiskBackedArray::new(data, file_path, strategy, read_only)
}

/// Load chunks of an out-of-core array into memory
#[allow(dead_code)]
pub fn load_chunks<A, D>(
    array: &OutOfCoreArray<A, D>,
    chunk_indices: &[usize],
) -> Result<Vec<Array<A, D>>, CoreError>
where
    A: Clone + Serialize + for<'de> Deserialize<'de>,
    D: Dimension + Serialize + for<'de> Deserialize<'de>,
{
    let num_chunks = array.num_chunks();

    // Validate chunk _indices
    for &idx in chunk_indices {
        if idx >= num_chunks {
            return Err(CoreError::IndexError(
                ErrorContext::new(format!("Chunk index out of bounds: {idx} >= {num_chunks}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }

    // If no chunks are requested, return empty vector
    if chunk_indices.is_empty() {
        return Ok(Vec::new());
    }

    // Sort chunk _indices to potentially optimize file reading
    // (this would matter more in a real implementation with proper chunk storage)
    let mut sorted_indices: Vec<usize> = chunk_indices.to_vec();
    sorted_indices.sort_unstable();

    // Remove duplicates
    sorted_indices.dedup();

    // Create a mapping from sorted _indices to original positions
    let mut index_map = Vec::with_capacity(chunk_indices.len());
    for &idx in chunk_indices {
        index_map.push(
            sorted_indices
                .iter()
                .position(|&x| x == idx)
                .expect("Operation failed"),
        );
    }

    // Load chunks in optimal order
    let mut sorted_chunks = Vec::with_capacity(sorted_indices.len());
    for &idx in &sorted_indices {
        sorted_chunks.push(array.load_chunk(idx)?);
    }

    // Rearrange chunks to match the original requested order
    let mut result = Vec::with_capacity(chunk_indices.len());
    for &pos in &index_map {
        result.push(sorted_chunks[pos].clone());
    }

    Ok(result)
}
