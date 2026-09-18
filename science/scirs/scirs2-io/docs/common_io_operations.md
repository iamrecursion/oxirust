# Common I/O Operations Tutorial

This tutorial demonstrates common input/output operations using the `scirs2-io` module, focusing on practical examples for scientific computing tasks.

## Table of Contents

1. [Reading and Writing Simple Data Files](#reading-and-writing-simple-data-files)
   - [CSV Files](#csv-files)
   - [Text-based Data](#text-based-data)
2. [Working with Scientific Data Formats](#working-with-scientific-data-formats)
   - [MATLAB Files](#matlab-files)
   - [ARFF Files](#arff-files)
3. [Memory-Efficient Processing of Large Files](#memory-efficient-processing-of-large-files)
   - [Chunked Reading](#chunked-reading)
   - [Streaming Processing](#streaming-processing)
4. [Data Serialization](#data-serialization)
   - [Array Serialization](#array-serialization)
   - [Structured Data](#structured-data)
5. [Data Compression](#data-compression)
   - [Compressing Files](#compressing-files)
   - [Compressing Arrays](#compressing-arrays)
6. [Data Validation](#data-validation)
   - [Checksum Generation](#checksum-generation)
   - [Format Validation](#format-validation)
7. [Multimedia File Handling](#multimedia-file-handling)
   - [Image Files](#image-files)
   - [Audio Files](#audio-files)

## Reading and Writing Simple Data Files

### CSV Files

CSV (Comma-Separated Values) is one of the most common formats for storing tabular data. The `scirs2-io` module provides comprehensive functionality for working with CSV files.

#### Basic Reading and Writing

```rust
use scirs2_io::csv::{read_csv, write_csv};
use ndarray::Array2;

// Read a CSV file with default settings
fn read_basic_csv() -> Result<(), Box<dyn std::error::Error>> {
    // Read CSV file, returns headers and data as an Array2<String>
    let (headers, data) = read_csv("data.csv", None)?;
    
    println!("Headers: {:?}", headers);
    println!("Data shape: {:?}", data.shape());
    println!("First row: {:?}", data.row(0));
    
    Ok(())
}

// Write a 2D array to a CSV file
fn write_basic_csv() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D array of floating-point values
    let data = Array2::from_shape_vec((3, 4), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    
    // Define column headers
    let headers = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    
    // Write to CSV file
    write_csv("output.csv", &data, Some(&headers), None)?;
    
    println!("CSV file written successfully");
    Ok(())
}
```

#### Type Conversion and Missing Value Handling

```rust
use scirs2_io::csv::{read_csv_typed, write_csv_typed, ColumnType, MissingValueOptions};
use std::collections::HashMap;

fn handle_typed_csv() -> Result<(), Box<dyn std::error::Error>> {
    // Define column specifications with types
    let mut column_types = HashMap::new();
    column_types.insert("id".to_string(), ColumnType::Integer);
    column_types.insert("name".to_string(), ColumnType::String);
    column_types.insert("value".to_string(), ColumnType::Float);
    column_types.insert("active".to_string(), ColumnType::Boolean);
    
    // Configure missing value handling
    let missing_values = MissingValueOptions::new()
        .with_na_strings(vec!["NA".to_string(), "N/A".to_string(), "".to_string()])
        .with_default_values(HashMap::from([
            (ColumnType::Integer, "0".to_string()),
            (ColumnType::Float, "0.0".to_string()),
            (ColumnType::Boolean, "false".to_string()),
            (ColumnType::String, "".to_string()),
        ]));
    
    // Read typed CSV
    let (headers, data) = read_csv_typed("data.csv", column_types, Some(missing_values), None)?;
    
    // Process data...
    
    Ok(())
}
```

#### Processing Large CSV Files

```rust
use scirs2_io::csv::{read_csv_chunked, CsvReaderConfig};

fn process_large_csv() -> Result<(), Box<dyn std::error::Error>> {
    // Configure CSV reader
    let config = CsvReaderConfig {
        delimiter: ',',
        has_header: true,
        ..Default::default()
    };
    
    // Process file in chunks
    let chunk_size = 1000; // Process 1000 rows at a time
    read_csv_chunked("large_file.csv", Some(config), chunk_size, |chunk, _| {
        // Process each chunk
        println!("Processing chunk with shape: {:?}", chunk.shape());
        
        // Do something with the chunk...
        
        // Continue processing
        Ok(true)
    })?;
    
    Ok(())
}
```

### Text-based Data

#### Reading and Writing Simple Text Files

```rust
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};

fn text_file_io() -> Result<(), Box<dyn std::error::Error>> {
    // Reading a text file
    let file = File::open("input.txt")?;
    let mut reader = BufReader::new(file);
    let mut content = String::new();
    reader.read_to_string(&mut content)?;
    
    // Process the content...
    println!("Read {} characters", content.len());
    
    // Writing to a text file
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("output.txt")?;
        
    let mut writer = BufWriter::new(file);
    writer.write_all(b"This is a sample text file.\n")?;
    writer.write_all(b"It contains multiple lines of text.\n")?;
    writer.flush()?;
    
    Ok(())
}
```

## Working with Scientific Data Formats

### MATLAB Files

MATLAB (.mat) files are commonly used for storing numerical data and variables in scientific computing.

```rust
use scirs2_io::matlab::{loadmat, MatFile};
use ndarray::Array2;

fn matlab_file_example() -> Result<(), Box<dyn std::error::Error>> {
    // Reading a MATLAB file
    let mat_data = loadmat("data.mat")?;
    
    // Access variables by name
    let x = mat_data.get_array::<f64>("x")?;
    println!("Variable x has shape: {:?}", x.shape());
    
    // Create a new MATLAB file
    let mut mat_file = MatFile::new();
    
    // Add variables
    let array = Array2::<f64>::zeros((3, 3));
    mat_file.add_array("zeros", &array)?;
    
    // Add a scalar
    mat_file.add_scalar("scalar", 42.0)?;
    
    // Save to file
    mat_file.save("output.mat")?;
    
    Ok(())
}
```

### ARFF Files

ARFF (Attribute-Relation File Format) is used by the Weka machine learning software.

```rust
use scirs2_io::arff::{load_arff, ArffDataset, ArffAttribute, ArffAttributeType};

fn arff_file_example() -> Result<(), Box<dyn std::error::Error>> {
    // Reading an ARFF file
    let dataset = load_arff("data.arff")?;
    
    println!("Dataset name: {}", dataset.relation);
    println!("Attributes: {:?}", dataset.attributes.iter().map(|attr| &attr.name).collect::<Vec<_>>());
    println!("Number of instances: {}", dataset.data.len());
    
    // Create a new ARFF dataset
    let mut new_dataset = ArffDataset::new("my_dataset");
    
    // Add attributes
    new_dataset.add_attribute(ArffAttribute {
        name: "numeric_attr".to_string(),
        attribute_type: ArffAttributeType::Numeric,
    })?;
    
    new_dataset.add_attribute(ArffAttribute {
        name: "nominal_attr".to_string(),
        attribute_type: ArffAttributeType::Nominal(vec!["class1".to_string(), "class2".to_string()]),
    })?;
    
    // Save the dataset
    new_dataset.save("output.arff")?;
    
    Ok(())
}
```

## Memory-Efficient Processing of Large Files

### Chunked Reading

Process large files efficiently by reading them in manageable chunks.

```rust
use scirs2_io::csv::read_csv_chunked;
use ndarray::Array2;

fn process_large_file() -> Result<(), Box<dyn std::error::Error>> {
    // Variables to accumulate statistics
    let mut total_rows = 0;
    let mut sum = 0.0;
    let mut count = 0;
    
    // Process a large CSV file in chunks
    read_csv_chunked("large_dataset.csv", None, 5000, |chunk, _headers| {
        // Update statistics
        total_rows += chunk.nrows();
        
        // Process numeric column (assuming column 1 is numeric)
        for i in 0..chunk.nrows() {
            if let Ok(val) = chunk[[i, 1]].parse::<f64>() {
                sum += val;
                count += 1;
            }
        }
        
        // Continue processing
        Ok(true)
    })?;
    
    // Calculate final statistics
    let average = if count > 0 { sum / count as f64 } else { 0.0 };
    println!("Processed {} rows", total_rows);
    println!("Average value: {:.2}", average);
    
    Ok(())
}
```

### Streaming Processing

Implement streaming data processing for efficient handling of large datasets.

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn streaming_text_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Open a large text file
    let file = File::open("large_log.txt")?;
    let reader = BufReader::new(file);
    
    // Process line by line without loading entire file into memory
    let mut line_count = 0;
    let mut error_count = 0;
    
    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        
        // Example: Count error lines
        if line.contains("ERROR") {
            error_count += 1;
        }
    }
    
    println!("Processed {} lines, found {} errors", line_count, error_count);
    
    Ok(())
}
```

## Data Serialization

### Array Serialization

```rust
use scirs2_io::serialize::{serialize_array, deserialize_array, SerializationFormat};
use ndarray::{Array2, ArrayD, IxDyn};

fn array_serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D array
    let array = Array2::from_shape_vec((3, 4), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    
    // Convert to dynamic-dimensional array for serialization
    let array_dyn = array.into_dyn();
    
    // Serialize to different formats
    serialize_array("array.bin", &array_dyn, SerializationFormat::Binary)?;
    serialize_array("array.json", &array_dyn, SerializationFormat::JSON)?;
    serialize_array("array.msgpack", &array_dyn, SerializationFormat::MessagePack)?;
    
    // Deserialize from binary
    let loaded_array: ArrayD<f64> = deserialize_array("array.bin", SerializationFormat::Binary)?;
    
    println!("Original shape: {:?}", array_dyn.shape());
    println!("Loaded shape: {:?}", loaded_array.shape());
    
    Ok(())
}
```

### Structured Data

```rust
use scirs2_io::serialize::{serialize_struct, deserialize_struct, SerializationFormat};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Experiment {
    id: u32,
    name: String,
    parameters: Vec<f64>,
    timestamp: u64,
    metadata: std::collections::HashMap<String, String>,
}

fn struct_serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample struct
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("author".to_string(), "Researcher".to_string());
    metadata.insert("lab".to_string(), "Physics Lab".to_string());
    
    let experiment = Experiment {
        id: 42,
        name: "Quantum Fluctuation Measurement".to_string(),
        parameters: vec![0.5, 1.2, 3.7, 4.2],
        timestamp: 1649876543,
        metadata,
    };
    
    // Serialize to different formats
    serialize_struct("experiment.json", &experiment, SerializationFormat::JSON)?;
    serialize_struct("experiment.bin", &experiment, SerializationFormat::Binary)?;
    
    // Deserialize 
    let loaded: Experiment = deserialize_struct("experiment.json", SerializationFormat::JSON)?;
    
    println!("Loaded experiment: {:?}", loaded);
    
    Ok(())
}
```

## Data Compression

### Compressing Files

```rust
use scirs2_io::compression::{
    compress_file, 
    decompress_file, 
    CompressionAlgorithm,
    compression_ratio,
};

fn file_compression_example() -> Result<(), Box<dyn std::error::Error>> {
    // Compress a file using different algorithms
    let file_path = "large_dataset.csv";
    
    // Try different compression algorithms
    for algorithm in &[
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Bzip2,
    ] {
        // Compress the file
        let compressed_path = compress_file(
            file_path, 
            None, // Auto-generate output path
            *algorithm, 
            Some(6) // Compression level
        )?;
        
        // Get file sizes
        let original_size = std::fs::metadata(file_path)?.len();
        let compressed_size = std::fs::metadata(&compressed_path)?.len();
        
        println!(
            "{:?} compression: {:.2} MB → {:.2} MB (ratio: {:.2}x)",
            algorithm,
            original_size as f64 / 1024.0 / 1024.0,
            compressed_size as f64 / 1024.0 / 1024.0,
            original_size as f64 / compressed_size as f64
        );
        
        // Decompress the file
        let decompressed_path = decompress_file(&compressed_path, None, Some(*algorithm))?;
        
        println!("Decompressed to: {}", decompressed_path);
        
        // Clean up
        std::fs::remove_file(compressed_path)?;
        std::fs::remove_file(decompressed_path)?;
    }
    
    Ok(())
}
```

### Compressing Arrays

```rust
use scirs2_io::compression::ndarray::{
    compress_array,
    decompress_array,
    compress_array_chunked,
    decompress_array_chunked,
};
use scirs2_io::compression::CompressionAlgorithm;
use ndarray::{Array, ArrayD, IxDyn, OwnedRepr};

fn array_compression_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large array with repeating patterns
    let shape = vec![100, 100, 10]; // 100,000 elements
    let mut data = Vec::with_capacity(shape.iter().product());
    
    // Fill with patterned data (good for compression)
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                data.push((i as f64 / 20.0).sin() + (j as f64 / 20.0).cos() + (k as f64 / 5.0).sin());
            }
        }
    }
    
    // Create ndarray
    let array = Array::from_shape_vec(IxDyn(&shape), data)?;
    
    // Compress using ZSTD
    let output_path = "compressed_array.zst";
    compress_array(output_path, &array, CompressionAlgorithm::Zstd, Some(6), None)?;
    
    // Get file size
    let file_size = std::fs::metadata(output_path)?.len();
    let original_size = array.len() * std::mem::size_of::<f64>();
    
    println!(
        "Array compression: {:.2} MB → {:.2} MB (ratio: {:.2}x)",
        original_size as f64 / 1024.0 / 1024.0,
        file_size as f64 / 1024.0 / 1024.0,
        original_size as f64 / file_size as f64
    );
    
    // Decompress
    let decompressed: ArrayD<f64> = decompress_array(output_path)?;
    
    // Verify first few elements
    for i in 0..5 {
        assert!((array[[i, 0, 0]] - decompressed[[i, 0, 0]]).abs() < 1e-10);
    }
    
    println!("Decompression verified successfully");
    
    // Clean up
    std::fs::remove_file(output_path)?;
    
    Ok(())
}
```

## Data Validation

### Checksum Generation

```rust
use scirs2_io::validation::{
    calculate_checksum,
    verify_checksum,
    calculate_file_checksum, 
    verify_file_checksum,
    ChecksumAlgorithm,
};

fn checksum_example() -> Result<(), Box<dyn std::error::Error>> {
    // Generate checksums for data
    let data = b"Important scientific data that needs verification";
    
    let crc32_checksum = calculate_checksum(data, ChecksumAlgorithm::CRC32);
    let sha256_checksum = calculate_checksum(data, ChecksumAlgorithm::SHA256);
    let blake3_checksum = calculate_checksum(data, ChecksumAlgorithm::BLAKE3);
    
    println!("CRC32: {}", crc32_checksum);
    println!("SHA-256: {}", sha256_checksum);
    println!("BLAKE3: {}", blake3_checksum);
    
    // Calculate file checksum
    let file_path = "large_dataset.csv";
    let file_checksum = calculate_file_checksum(file_path, ChecksumAlgorithm::SHA256)?;
    
    println!("File checksum: {}", file_checksum);
    
    // Verify integrity
    let is_valid = verify_file_checksum(file_path, &file_checksum, ChecksumAlgorithm::SHA256)?;
    println!("File integrity valid: {}", is_valid);
    
    Ok(())
}
```

### Format Validation

```rust
use scirs2_io::validation::formats::{
    validate_format,
    detect_file_format,
    validate_file_format,
    DataFormat,
};
use std::path::Path;

fn format_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = Path::new("data.csv");
    let json_path = Path::new("config.json");
    
    // Detect file formats
    if let Some(format) = detect_file_format(csv_path)? {
        println!("Detected format for CSV file: {}", format);
    }
    
    if let Some(format) = detect_file_format(json_path)? {
        println!("Detected format for JSON file: {}", format);
    }
    
    // Validate specific formats
    let csv_valid = validate_format(csv_path, DataFormat::CSV)?;
    println!("CSV format valid: {}", csv_valid);
    
    // Detailed validation
    let result = validate_file_format(csv_path, DataFormat::CSV)?;
    println!("CSV validation: valid={}, details={:?}", result.valid, result.details);
    
    Ok(())
}
```

## Multimedia File Handling

### Image Files

```rust
use scirs2_io::image::{
    read_image,
    write_image,
    read_image_metadata,
    convert_image,
    get_grayscale,
    ImageFormat,
    ColorMode,
};
use ndarray::Array3;

fn image_processing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Read an image
    let (img_data, color_mode) = read_image("image.png")?;
    
    println!("Image shape: {:?}", img_data.shape());
    println!("Color mode: {:?}", color_mode);
    
    // Read metadata
    let metadata = read_image_metadata("image.png")?;
    println!("Image metadata: {:?}", metadata);
    
    // Convert to grayscale
    let grayscale = get_grayscale(&img_data, color_mode)?;
    println!("Grayscale shape: {:?}", grayscale.shape());
    
    // Save grayscale image
    write_image("grayscale.png", &grayscale, ColorMode::Grayscale, ImageFormat::PNG)?;
    
    // Convert between formats
    convert_image("image.png", "image.jpg", ImageFormat::JPEG)?;
    
    Ok(())
}
```

### Audio Files

```rust
use scirs2_io::wavfile::{read_wav, write_wav};
use ndarray::Array1;

fn audio_processing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Read a WAV file
    let (rate, data) = read_wav("audio.wav")?;
    
    println!("Audio: {} Hz, {} samples", rate, data.len());
    
    // Process audio data
    // Example: increase volume by 2x
    let amplified: Vec<i16> = data.iter()
        .map(|&sample| {
            // Prevent overflow by clamping
            let sample_f32 = sample as f32 * 2.0;
            if sample_f32 > 32767.0 {
                32767
            } else if sample_f32 < -32768.0 {
                -32768
            } else {
                sample_f32 as i16
            }
        })
        .collect();
    
    // Write processed audio
    write_wav("amplified.wav", rate, &amplified)?;
    
    // Generate a sine wave
    let duration = 2.0; // seconds
    let freq = 440.0;   // Hz (A4 note)
    
    let samples = (rate as f64 * duration) as usize;
    let mut sine_data = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f64 / rate as f64;
        let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
        // Scale to i16 range
        sine_data.push((sample * 32767.0) as i16);
    }
    
    // Write the sine wave
    write_wav("sine_440hz.wav", rate, &sine_data)?;
    
    Ok(())
}
```

## Best Practices for Scientific Data I/O

When working with scientific data, consider the following best practices:

1. **Choose the right format**: Select the appropriate format based on your data type, size, and intended use.
   - For tabular data: CSV, HDF5
   - For matrices and variables: MATLAB files
   - For metadata-rich datasets: ARFF
   - For large arrays: Use compression and chunked processing

2. **Validate your data**: Always validate imported data and verify the integrity of exported data.
   - Use checksums for data verification
   - Validate file formats
   - Check data ranges and consistency

3. **Optimize for memory usage**: When dealing with large datasets:
   - Use chunked reading
   - Process data in streams
   - Consider compression for storage

4. **Handle missing values appropriately**: Scientific data often contains missing values.
   - Specify missing value markers
   - Provide default values or strategies for handling missing data

5. **Document your data formats**: When creating custom formats or serializing data:
   - Include metadata
   - Document the structure
   - Version your formats

By following these practices, you can ensure robust, efficient, and reliable data handling in your scientific computing applications.