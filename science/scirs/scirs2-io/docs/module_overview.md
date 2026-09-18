# scirs2-io Module Overview

The `scirs2-io` module provides comprehensive input/output functionality for scientific computing in Rust, modeled after SciPy's IO facilities but with Rust's safety and performance characteristics.

## Module Purpose

The primary purpose of `scirs2-io` is to provide scientific and numeric I/O functionality covering:

1. **Scientific data formats** - Reading and writing domain-specific formats like MATLAB, ARFF
2. **General-purpose formats** - Working with CSV, JSON, image files, and audio files
3. **Data serialization** - Efficiently serializing numerical data and structured objects
4. **Data compression** - Efficient storage and transmission of scientific data
5. **Data validation** - Ensuring data integrity and format compliance

## Architecture

The module is organized into several submodules, each handling a specific aspect of I/O operations:

```
scirs2-io
├── arff      - ARFF file format support
├── compression - Data compression utilities
├── csv       - CSV and delimited text file utilities
├── error     - Common error types and handling
├── image     - Image file support
├── matlab    - MATLAB .mat file support
├── serialize - Data serialization utilities
├── validation - Data validation and verification
└── wavfile   - WAV audio file support
```

### Error Handling

The module uses a consistent error handling approach with:

- A central `IoError` enum in the `error` module
- Error variants for different failure types (FileError, FormatError, etc.)
- Use of the `Result<T, IoError>` type for all operations that can fail

### Dependencies

The module leverages several Rust crates for specific functionality:

- `ndarray` - Core array representation
- `serde` - Serialization/deserialization framework
- `image` - Image format support
- `bincode`, `serde_json`, `rmp-serde` - Serialization formats
- `flate2`, `zstd`, `lz4`, `bzip2` - Compression algorithms
- `sha2`, `crc32fast`, `blake3` - Checksum algorithms

## Core Functionality

### CSV Module

The CSV module provides functionality for reading and writing comma-separated values and other delimited text files. Features include:

- Basic CSV reading and writing
- Type conversion and detection
- Missing value handling
- Memory-efficient processing of large files using chunked reading
- Support for custom delimiters, headers, and escape characters

Key types:
- `CsvReaderConfig` - Configuration for CSV reading
- `CsvWriterConfig` - Configuration for CSV writing
- `ColumnType` - Enum for column data types
- `DataValue` - Enum for mixed data types

Key functions:
- `read_csv` - Read CSV as string data
- `read_csv_numeric` - Read CSV as numeric data
- `read_csv_typed` - Read CSV with type conversion
- `read_csv_chunked` - Process large CSV files in chunks
- `write_csv` - Write 2D array to CSV
- `write_csv_typed` - Write mixed data types to CSV

### MATLAB Module

The MATLAB module supports reading and writing MATLAB .mat format files (Level 5 MAT-File Format).

Key types:
- `MatFile` - Representation of a MATLAB .mat file
- `MatVar` - Representation of MATLAB variables

Key functions:
- `loadmat` - Load variables from a .mat file
- `savemat` - Save variables to a .mat file

### ARFF Module

The ARFF (Attribute-Relation File Format) module provides support for Weka's ARFF format.

Key types:
- `ArffDataset` - Complete ARFF dataset
- `ArffAttribute` - ARFF attribute definition
- `ArffAttributeType` - Attribute types (Numeric, String, Nominal, Date)
- `ArffValue` - Attribute values (Numeric, String, Date, Nominal, Missing)

Key functions:
- `load_arff` - Load an ARFF file
- `save_arff` - Save an ARFF dataset to a file
- `get_numeric_matrix` - Extract numeric attributes as a matrix
- `numeric_matrix_to_arff` - Convert numeric matrix to ARFF

### WAV File Module

The WAV module provides functionality for reading and writing WAV audio files.

Key functions:
- `read_wav` - Read WAV file data
- `read_wav_info` - Read WAV file information
- `write_wav` - Write WAV file

Key types:
- `WavInfo` - WAV file information

### Image Module

The image module provides functionality for reading, writing, and basic processing of image files.

Key types:
- `ColorMode` - Image color mode (Grayscale, RGB, RGBA)
- `ImageFormat` - Image format (PNG, JPEG, BMP, TIFF)
- `ImageMetadata` - Image metadata

Key functions:
- `read_image` - Read image file
- `write_image` - Write image file
- `read_image_metadata` - Read image metadata
- `convert_image` - Convert between image formats
- `get_grayscale` - Convert image to grayscale

### Serialization Module

The serialization module provides utilities for serializing and deserializing scientific data.

Key types:
- `SerializationFormat` - Format (Binary, JSON, MessagePack)
- `SerializedArray` - Array with metadata for serialization
- `ArrayMetadata` - Metadata for array serialization
- `SparseMatrixCOO` - Sparse matrix in COO format

Key functions:
- `serialize_array` - Serialize ndarray
- `deserialize_array` - Deserialize ndarray
- `serialize_array_with_metadata` - Serialize with metadata
- `deserialize_array_with_metadata` - Deserialize with metadata
- `serialize_struct` - Serialize struct
- `deserialize_struct` - Deserialize struct
- `serialize_sparse_matrix` - Serialize sparse matrix
- `deserialize_sparse_matrix` - Deserialize sparse matrix

### Compression Module

The compression module provides utilities for compressing and decompressing data.

Key types:
- `CompressionAlgorithm` - Compression algorithm (Gzip, Zstd, Lz4, Bzip2)
- `CompressionInfo` - Information about algorithms

Key functions:
- `compress_data` - Compress raw data
- `decompress_data` - Decompress data
- `compress_file` - Compress a file
- `decompress_file` - Decompress a file
- `compression_ratio` - Calculate compression ratio
- `algorithm_info` - Get algorithm information

Ndarray submodule:
- `compress_array` - Compress ndarray
- `decompress_array` - Decompress ndarray
- `compress_array_chunked` - Chunk-wise compression
- `decompress_array_chunked` - Chunk-wise decompression
- `compare_compression_algorithms` - Benchmark algorithms

### Validation Module

The validation module provides utilities for data validation and integrity checking.

Key types:
- `ChecksumAlgorithm` - Checksum algorithm (CRC32, SHA256, BLAKE3)
- `IntegrityMetadata` - File integrity metadata
- `ValidationReport` - Validation result report
- `DirectoryManifest` - Directory validation manifest

Key functions:
- `calculate_checksum` - Calculate data checksum
- `verify_checksum` - Verify data integrity
- `calculate_file_checksum` - Calculate file checksum
- `verify_file_checksum` - Verify file integrity
- `generate_file_integrity_metadata` - Generate integrity metadata
- `validate_file_integrity` - Validate file against metadata
- `create_directory_manifest` - Create directory manifest
- `validate_file_format` - Validate file format

## Usage Patterns

### Basic File Operations

```rust
// Read a CSV file
let (headers, data) = scirs2_io::csv::read_csv("data.csv", None)?;

// Read an image file
let (image_data, color_mode) = scirs2_io::image::read_image("image.png")?;

// Read a MATLAB file
let mat_data = scirs2_io::matlab::loadmat("data.mat")?;
```

### Working with Large Files

```rust
// Process a large CSV file in chunks
scirs2_io::csv::read_csv_chunked("large_file.csv", None, 1000, |chunk, _| {
    // Process each chunk
    println!("Processing chunk with shape: {:?}", chunk.shape());
    Ok(true) // Continue processing
})?;
```

### Data Serialization

```rust
// Serialize array to different formats
serialize_array("data.bin", &array_dyn, SerializationFormat::Binary)?;
serialize_array("data.json", &array_dyn, SerializationFormat::JSON)?;
serialize_array("data.msgpack", &array_dyn, SerializationFormat::MessagePack)?;

// Deserialize from a file
let loaded: ArrayD<f64> = deserialize_array("data.bin", SerializationFormat::Binary)?;
```

### Data Compression

```rust
// Compress a file
let compressed_path = compress_file(
    "large_file.csv", 
    None,
    CompressionAlgorithm::Zstd,
    Some(6) // Compression level
)?;

// Decompress a file
let original_path = decompress_file(&compressed_path, None, Some(CompressionAlgorithm::Zstd))?;
```

### Data Validation

```rust
// Calculate file checksum
let checksum = calculate_file_checksum("data.bin", ChecksumAlgorithm::SHA256)?;

// Validate file integrity
let metadata = generate_file_integrity_metadata("data.bin", ChecksumAlgorithm::SHA256)?;
let is_valid = validate_file_integrity("data.bin", &metadata)?;

// Validate file format
let format_valid = validate_file_format("data.csv", DataFormat::CSV)?;
```

## Integration with other modules

The `scirs2-io` module is designed to work seamlessly with other modules in the SciRS2 ecosystem:

1. **Core integration** - Uses types and utilities from `scirs2-core`
2. **Linear algebra** - Works with `ndarray` types used in `scirs2-linalg`
3. **Statistics** - I/O for statistical data used in `scirs2-stats`
4. **Machine learning** - Data loading/saving for `scirs2-neural`

## Extension Points

The module is designed for extension:

1. **New formats** - Add support for additional formats (HDF5, NetCDF, etc.)
2. **Enhanced functionality** - Add features to existing formats
3. **Optimization** - Performance improvements for specific formats
4. **Domain-specific I/O** - Add specialized I/O for specific scientific domains

## Performance Considerations

The module is optimized for scientific computing workloads:

1. **Memory efficiency** - Chunked processing for large datasets
2. **Speed** - Fast serialization/deserialization
3. **Size** - Compression for efficient storage
4. **Validation** - Integrity checking for data integrity

## Best Practices

When using the `scirs2-io` module, consider these best practices:

1. **Choose appropriate formats** - Match the format to your data type and usage
2. **Handle errors** - Always check for errors and handle them gracefully
3. **Validate data** - Verify data integrity, especially for important data
4. **Use compression** - For large datasets, consider compression
5. **Document formats** - Document the file formats you use
6. **Test I/O code** - Thoroughly test I/O operations
7. **Follow conventions** - Use standard file extensions and formats

## Examples

See the following files for detailed examples:

- [Common I/O Operations](common_io_operations.md)
- [File Format Examples](file_format_examples.md)

## Future Directions

The module roadmap includes:

1. **HDF5 support** - Reading and writing HDF5 files
2. **NetCDF support** - Reading and writing NetCDF files
3. **Enhanced image support** - Image sequence handling
4. **Network data exchange** - Data transfer protocols and remote data access
5. **Cloud storage integration** - Support for cloud storage systems
6. **Domain-specific I/O** - Specialized I/O for various scientific fields