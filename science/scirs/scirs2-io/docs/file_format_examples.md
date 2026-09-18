# File Format Examples

This document provides detailed examples for working with various file formats supported by the `scirs2-io` module.

## Table of Contents

1. [CSV Files](#csv-files)
2. [MATLAB .mat Files](#matlab-mat-files)
3. [ARFF Files](#arff-files)
4. [WAV Audio Files](#wav-audio-files)
5. [Image Files](#image-files)
6. [JSON Serialization](#json-serialization)
7. [Binary Serialization](#binary-serialization)
8. [MessagePack Serialization](#messagepack-serialization)

## CSV Files

CSV (Comma-Separated Values) files are widely used for tabular data storage and exchange.

### Basic CSV Reading

```rust
use scirs2_io::csv::{read_csv, CsvReaderConfig};
use ndarray::Array2;

fn read_csv_example() -> Result<(), Box<dyn std::error::Error>> {
    // Default configuration
    let (headers, data) = read_csv("data.csv", None)?;
    
    println!("Headers: {:?}", headers);
    println!("Data dimensions: {:?}", data.shape());
    
    // With custom configuration
    let config = CsvReaderConfig {
        delimiter: ';',
        has_header: true,
        comment_char: Some('#'),
        quote_char: Some('"'),
        escape_char: Some('\\'),
        ..Default::default()
    };
    
    let (headers, data) = read_csv("semicolon_data.csv", Some(config))?;
    
    // Access data elements
    for i in 0..data.nrows().min(5) {
        println!("Row {}: {:?}", i, data.row(i));
    }
    
    Ok(())
}
```

### CSV with Type Conversion

```rust
use scirs2_io::csv::{read_csv_typed, ColumnType, MissingValueOptions};
use scirs2_io::csv::DataValue;
use std::collections::HashMap;

fn read_typed_csv() -> Result<(), Box<dyn std::error::Error>> {
    // Define column types
    let mut column_types = HashMap::new();
    column_types.insert("id".to_string(), ColumnType::Integer);
    column_types.insert("name".to_string(), ColumnType::String);
    column_types.insert("score".to_string(), ColumnType::Float);
    column_types.insert("active".to_string(), ColumnType::Boolean);
    
    // Configure missing value handling
    let missing_values = MissingValueOptions::new()
        .with_na_strings(vec!["NA".to_string(), "N/A".to_string()])
        .with_default_values(HashMap::from([
            (ColumnType::Integer, "0".to_string()),
            (ColumnType::Float, "0.0".to_string()),
            (ColumnType::Boolean, "false".to_string()),
            (ColumnType::String, "".to_string()),
        ]));
    
    // Read with type conversion
    let (headers, data) = read_csv_typed(
        "data.csv", 
        column_types.clone(), 
        Some(missing_values), 
        None
    )?;
    
    // Process the strongly-typed data
    for row in 0..data.len().min(5) {
        // Access row elements with proper types
        let id = match &data[row][0] {
            DataValue::Integer(val) => *val,
            _ => panic!("Expected integer"),
        };
        
        let name = match &data[row][1] {
            DataValue::String(val) => val.clone(),
            _ => panic!("Expected string"),
        };
        
        let score = match &data[row][2] {
            DataValue::Float(val) => *val,
            _ => panic!("Expected float"),
        };
        
        println!("ID: {}, Name: {}, Score: {:.1}", id, name, score);
    }
    
    Ok(())
}
```

### Writing CSV Files

```rust
use scirs2_io::csv::{write_csv, write_csv_typed, CsvWriterConfig};
use ndarray::Array2;
use scirs2_io::csv::DataValue;

fn write_csv_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D array
    let data = Array2::from_shape_vec((3, 4), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    
    // Define column headers
    let headers = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    
    // Default CSV writer
    write_csv("output1.csv", &data, Some(&headers), None)?;
    
    // Custom configuration
    let config = CsvWriterConfig {
        delimiter: ';',
        line_ending: "\r\n".to_string(),
        always_quote: true,
        ..Default::default()
    };
    
    write_csv("output2.csv", &data, Some(&headers), Some(config))?;
    
    // Write typed data
    let typed_data = vec![
        vec![
            DataValue::Integer(1),
            DataValue::String("Alice".to_string()),
            DataValue::Float(93.5),
            DataValue::Boolean(true),
        ],
        vec![
            DataValue::Integer(2),
            DataValue::String("Bob".to_string()),
            DataValue::Float(85.0),
            DataValue::Boolean(false),
        ],
    ];
    
    let typed_headers = vec![
        "id".to_string(),
        "name".to_string(),
        "score".to_string(),
        "active".to_string(),
    ];
    
    write_csv_typed("typed_output.csv", &typed_data, Some(&typed_headers), None)?;
    
    Ok(())
}
```

### Chunked CSV Reading for Large Files

```rust
use scirs2_io::csv::{read_csv_chunked, CsvReaderConfig};
use std::io::Write;

fn process_large_csv() -> Result<(), Box<dyn std::error::Error>> {
    let mut output_file = std::fs::File::create("summary.txt")?;
    
    // Set up accumulators for statistics
    let mut total_rows = 0;
    let mut column_sums = Vec::new();
    let mut column_counts = Vec::new();
    
    // Process the CSV file in chunks
    read_csv_chunked("large_file.csv", None, 1000, |chunk, headers| {
        // Initialize column statistics if first chunk
        if column_sums.is_empty() {
            column_sums = vec![0.0; chunk.ncols()];
            column_counts = vec![0; chunk.ncols()];
            
            // Write headers to output file
            if let Some(headers) = headers {
                writeln!(output_file, "Statistics for {}", "large_file.csv")?;
                writeln!(output_file, "Columns: {:?}", headers)?;
            }
        }
        
        // Update row count
        total_rows += chunk.nrows();
        
        // Process numeric columns
        for col in 0..chunk.ncols() {
            for row in 0..chunk.nrows() {
                if let Ok(value) = chunk[[row, col]].parse::<f64>() {
                    column_sums[col] += value;
                    column_counts[col] += 1;
                }
            }
        }
        
        // Report progress (optional)
        println!("Processed {} rows so far", total_rows);
        
        // Continue processing
        Ok(true)
    })?;
    
    // Calculate and write final statistics
    writeln!(output_file, "Total rows: {}", total_rows)?;
    writeln!(output_file, "Column averages:")?;
    
    for col in 0..column_sums.len() {
        let avg = if column_counts[col] > 0 {
            column_sums[col] / column_counts[col] as f64
        } else {
            0.0
        };
        
        writeln!(output_file, "  Column {}: {:.2}", col, avg)?;
    }
    
    Ok(())
}
```

## MATLAB .mat Files

MATLAB .mat files store numerical arrays, matrices, and other MATLAB variables.

### Reading MATLAB Files

```rust
use scirs2_io::matlab::loadmat;
use ndarray::Array2;

fn read_matlab_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load the .mat file
    let mat_data = loadmat("data.mat")?;
    
    // List all variables in the file
    println!("Variables in the file:");
    for (name, _) in &mat_data.variables {
        println!("  {}", name);
    }
    
    // Get a specific variable as an array
    if let Some(x_var) = mat_data.variables.get("x") {
        // Convert to ndarray (assuming it's a 2D array of doubles)
        let x_array = x_var.to_array::<f64, _>()?;
        
        println!("Variable 'x' shape: {:?}", x_array.shape());
        
        // Access elements
        if x_array.shape()[0] > 0 && x_array.shape()[1] > 0 {
            println!("First element: {}", x_array[[0, 0]]);
        }
    }
    
    // Get a scalar value
    if let Some(scalar_var) = mat_data.variables.get("scalar") {
        let scalar_value = scalar_var.to_scalar::<f64>()?;
        println!("Scalar value: {}", scalar_value);
    }
    
    Ok(())
}
```

### Writing MATLAB Files

```rust
use scirs2_io::matlab::{MatFile, MatVar};
use ndarray::{Array1, Array2, Array3};

fn write_matlab_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new MatFile
    let mut mat_file = MatFile::new();
    
    // Add a 1D array (vector)
    let vector = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    mat_file.add_array("vector", &vector)?;
    
    // Add a 2D array (matrix)
    let matrix = Array2::<f64>::from_shape_vec((3, 3), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;
    mat_file.add_array("matrix", &matrix)?;
    
    // Add a 3D array
    let array3d = Array3::<f64>::from_shape_fn((2, 2, 2), |(i, j, k)| {
        (i * 4 + j * 2 + k) as f64
    });
    mat_file.add_array("array3d", &array3d)?;
    
    // Add scalar values
    mat_file.add_scalar("int_scalar", 42)?;
    mat_file.add_scalar("float_scalar", 3.14159)?;
    
    // Add a string
    mat_file.add_string("string_var", "Hello, MATLAB!")?;
    
    // Save the file
    mat_file.save("output.mat")?;
    
    println!("MATLAB file saved successfully");
    
    Ok(())
}
```

### Working with Complex MATLAB Data

```rust
use scirs2_io::matlab::{loadmat, MatFile, MatVar};
use ndarray::Array2;
use std::collections::HashMap;

fn complex_matlab_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a structured data type for MATLAB
    let mut mat_file = MatFile::new();
    
    // Create a 2D array
    let data = Array2::<f64>::from_shape_fn((4, 3), |(i, j)| {
        (i * 3 + j) as f64 * 1.5
    });
    
    // Add it to the file
    mat_file.add_array("data", &data)?;
    
    // Create a structure
    let mut struct_fields = HashMap::new();
    
    // Add fields to the structure
    struct_fields.insert("name".to_string(), MatVar::String("Experiment".to_string()));
    struct_fields.insert("value".to_string(), MatVar::Scalar(42.0));
    struct_fields.insert("enabled".to_string(), MatVar::Scalar(1.0)); // Boolean as 1.0
    
    // Add structure to the file
    mat_file.add_struct("config", struct_fields)?;
    
    // Save the file
    mat_file.save("complex_data.mat")?;
    
    // Reading the complex data back
    let loaded = loadmat("complex_data.mat")?;
    
    // Access the structure
    if let Some(config_var) = loaded.variables.get("config") {
        if let MatVar::Struct(fields) = config_var {
            println!("Structure fields:");
            
            for (key, value) in fields {
                match value {
                    MatVar::String(s) => println!("  {}: \"{}\"", key, s),
                    MatVar::Scalar(v) => println!("  {}: {}", key, v),
                    _ => println!("  {}: <complex type>", key),
                }
            }
        }
    }
    
    Ok(())
}
```

## ARFF Files

ARFF (Attribute-Relation File Format) files are used by the Weka machine learning software.

### Reading ARFF Files

```rust
use scirs2_io::arff::{load_arff, ArffValue};

fn read_arff_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load an ARFF file
    let dataset = load_arff("data.arff")?;
    
    // Display dataset information
    println!("Dataset: {}", dataset.relation);
    println!("Number of attributes: {}", dataset.attributes.len());
    println!("Number of instances: {}", dataset.data.len());
    
    // Display attribute information
    println!("Attributes:");
    for (i, attr) in dataset.attributes.iter().enumerate() {
        print!("  {}: {} (", i, attr.name);
        
        match &attr.attribute_type {
            scirs2_io::arff::ArffAttributeType::Numeric => println!("numeric)"),
            scirs2_io::arff::ArffAttributeType::String => println!("string)"),
            scirs2_io::arff::ArffAttributeType::Date(format) => println!("date, format: {}))", format),
            scirs2_io::arff::ArffAttributeType::Nominal(values) => {
                print!("nominal, values: [");
                for (j, val) in values.iter().enumerate() {
                    if j > 0 { print!(", "); }
                    print!("{}", val);
                }
                println!("])");
            }
        }
    }
    
    // Display a few instances
    println!("First 5 instances:");
    for (i, instance) in dataset.data.iter().enumerate().take(5) {
        println!("  Instance {}:", i);
        
        for (j, value) in instance.iter().enumerate() {
            let attr_name = &dataset.attributes[j].name;
            match value {
                ArffValue::Numeric(val) => println!("    {}: {}", attr_name, val),
                ArffValue::String(val) => println!("    {}: \"{}\"", attr_name, val),
                ArffValue::Date(val) => println!("    {}: {}", attr_name, val),
                ArffValue::Nominal(val) => println!("    {}: {}", attr_name, val),
                ArffValue::Missing => println!("    {}: ?", attr_name),
            }
        }
    }
    
    Ok(())
}
```

### Creating ARFF Files

```rust
use scirs2_io::arff::{ArffDataset, ArffAttribute, ArffAttributeType, ArffValue};

fn create_arff_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new dataset
    let mut dataset = ArffDataset::new("iris_dataset");
    
    // Add relation comment
    dataset.add_relation_comment("Iris flower dataset");
    
    // Define attributes
    dataset.add_attribute(ArffAttribute {
        name: "sepal_length".to_string(),
        attribute_type: ArffAttributeType::Numeric,
    })?;
    
    dataset.add_attribute(ArffAttribute {
        name: "sepal_width".to_string(),
        attribute_type: ArffAttributeType::Numeric,
    })?;
    
    dataset.add_attribute(ArffAttribute {
        name: "petal_length".to_string(),
        attribute_type: ArffAttributeType::Numeric,
    })?;
    
    dataset.add_attribute(ArffAttribute {
        name: "petal_width".to_string(),
        attribute_type: ArffAttributeType::Numeric,
    })?;
    
    dataset.add_attribute(ArffAttribute {
        name: "class".to_string(),
        attribute_type: ArffAttributeType::Nominal(vec![
            "Iris-setosa".to_string(),
            "Iris-versicolor".to_string(),
            "Iris-virginica".to_string(),
        ]),
    })?;
    
    // Add data instances
    // Iris-setosa
    dataset.add_instance(vec![
        ArffValue::Numeric(5.1),
        ArffValue::Numeric(3.5),
        ArffValue::Numeric(1.4),
        ArffValue::Numeric(0.2),
        ArffValue::Nominal("Iris-setosa".to_string()),
    ])?;
    
    dataset.add_instance(vec![
        ArffValue::Numeric(4.9),
        ArffValue::Numeric(3.0),
        ArffValue::Numeric(1.4),
        ArffValue::Numeric(0.2),
        ArffValue::Nominal("Iris-setosa".to_string()),
    ])?;
    
    // Iris-versicolor
    dataset.add_instance(vec![
        ArffValue::Numeric(7.0),
        ArffValue::Numeric(3.2),
        ArffValue::Numeric(4.7),
        ArffValue::Numeric(1.4),
        ArffValue::Nominal("Iris-versicolor".to_string()),
    ])?;
    
    // Iris-virginica
    dataset.add_instance(vec![
        ArffValue::Numeric(6.3),
        ArffValue::Numeric(3.3),
        ArffValue::Numeric(6.0),
        ArffValue::Numeric(2.5),
        ArffValue::Nominal("Iris-virginica".to_string()),
    ])?;
    
    // Save the dataset
    dataset.save("iris.arff")?;
    
    println!("ARFF file created successfully");
    
    Ok(())
}
```

### Converting Between ARFF and Numeric Matrix

```rust
use scirs2_io::arff::{load_arff, get_numeric_matrix, numeric_matrix_to_arff};
use ndarray::Array2;

fn arff_matrix_conversion() -> Result<(), Box<dyn std::error::Error>> {
    // Load an ARFF file
    let dataset = load_arff("data.arff")?;
    
    // Convert to numeric matrix (only numeric attributes)
    let (matrix, attribute_indices, class_mapping) = get_numeric_matrix(&dataset)?;
    
    println!("Extracted {} numeric attributes", attribute_indices.len());
    println!("Matrix shape: {:?}", matrix.shape());
    
    if let Some(mapping) = class_mapping {
        println!("Class mapping:");
        for (nominal_value, numeric_value) in mapping {
            println!("  {} -> {}", nominal_value, numeric_value);
        }
    }
    
    // Create a new ARFF file from numeric matrix
    let new_dataset = numeric_matrix_to_arff(
        "converted_dataset",
        &matrix,
        &attribute_indices,
        &dataset,
        None,
    )?;
    
    // Save the new dataset
    new_dataset.save("converted.arff")?;
    
    println!("Converted ARFF file saved successfully");
    
    Ok(())
}
```

## WAV Audio Files

WAV files are a common format for storing audio data.

### Reading WAV Files

```rust
use scirs2_io::wavfile::{read_wav, read_wav_info};

fn read_wav_example() -> Result<(), Box<dyn std::error::Error>> {
    // Get information about the WAV file
    let info = read_wav_info("audio.wav")?;
    
    println!("WAV File Info:");
    println!("  Sample rate: {} Hz", info.rate);
    println!("  Channels: {}", info.channels);
    println!("  Sample width: {} bytes", info.sample_width);
    println!("  Number of frames: {}", info.nframes);
    println!("  Compression type: {}", info.compression_type);
    println!("  Compression name: {}", info.compression_name);
    
    // Read the actual audio data
    let (rate, data) = read_wav("audio.wav")?;
    
    println!("Sample rate: {} Hz", rate);
    println!("Number of samples: {}", data.len());
    
    // Calculate audio duration
    let duration = data.len() as f64 / rate as f64;
    println!("Duration: {:.2} seconds", duration);
    
    // Basic audio analysis
    if !data.is_empty() {
        let max_amplitude = data.iter().map(|&s| s.abs() as i32).max().unwrap_or(0);
        let min_amplitude = data.iter().map(|&s| s.abs() as i32).min().unwrap_or(0);
        
        println!("Maximum amplitude: {} ({:.2}% of max)", 
            max_amplitude, max_amplitude as f64 * 100.0 / 32768.0);
        println!("Minimum amplitude: {} ({:.2}% of max)", 
            min_amplitude, min_amplitude as f64 * 100.0 / 32768.0);
    }
    
    Ok(())
}
```

### Writing WAV Files

```rust
use scirs2_io::wavfile::write_wav;
use std::f64::consts::PI;

fn write_wav_example() -> Result<(), Box<dyn std::error::Error>> {
    // Generate audio data - a simple sine wave
    let rate = 44100; // 44.1 kHz sample rate
    let duration = 5.0; // 5 seconds
    let frequency = 440.0; // A4 note (440 Hz)
    
    let samples = (rate as f64 * duration) as usize;
    let mut audio_data = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f64 / rate as f64;
        let sample = (2.0 * PI * frequency * t).sin();
        
        // Convert to i16 range (-32768 to 32767)
        let sample_i16 = (sample * 32767.0) as i16;
        audio_data.push(sample_i16);
    }
    
    // Write to WAV file
    write_wav("sine_440hz.wav", rate, &audio_data)?;
    
    println!("WAV file created successfully");
    
    // Create a more complex waveform - a chord (A major: A, C#, E)
    let frequencies = [440.0, 554.37, 659.25]; // A4, C#5, E5
    let mut chord_data = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f64 / rate as f64;
        
        // Sum the sine waves and normalize
        let sample = frequencies.iter()
            .map(|&freq| (2.0 * PI * freq * t).sin())
            .sum::<f64>() / frequencies.len() as f64;
        
        // Apply amplitude envelope (fade in/out)
        let envelope = if t < 0.1 {
            t / 0.1 // Fade in
        } else if t > duration - 0.1 {
            (duration - t) / 0.1 // Fade out
        } else {
            1.0 // Sustain
        };
        
        // Convert to i16 range with envelope
        let sample_i16 = (sample * envelope * 32767.0) as i16;
        chord_data.push(sample_i16);
    }
    
    // Write to WAV file
    write_wav("a_major_chord.wav", rate, &chord_data)?;
    
    println!("Chord WAV file created successfully");
    
    Ok(())
}
```

### Processing WAV Files

```rust
use scirs2_io::wavfile::{read_wav, write_wav};

fn process_wav_example() -> Result<(), Box<dyn std::error::Error>> {
    // Read the input WAV file
    let (rate, data) = read_wav("input.wav")?;
    
    // Process the audio - apply a simple low-pass filter
    let mut filtered_data = Vec::with_capacity(data.len());
    
    // Simple moving average filter (very basic low-pass)
    let window_size = 5;
    for i in 0..data.len() {
        let mut sum = 0;
        let mut count = 0;
        
        for j in 0..window_size {
            if i >= j {
                sum += data[i - j] as i32;
                count += 1;
            }
        }
        
        let avg = if count > 0 { sum / count } else { 0 };
        filtered_data.push(avg as i16);
    }
    
    // Write the processed audio
    write_wav("filtered.wav", rate, &filtered_data)?;
    
    println!("Audio processing complete");
    
    // Create a stereo file from a mono file
    if data.len() > 0 {
        let mut stereo_data = Vec::with_capacity(data.len() * 2);
        
        for sample in &data {
            // Left channel
            stereo_data.push(*sample);
            // Right channel (inverted phase for demonstration)
            stereo_data.push(-*sample);
        }
        
        // Write stereo file
        let stereo_rate = rate;
        write_wav("stereo.wav", stereo_rate, &stereo_data)?;
        
        println!("Stereo WAV file created");
    }
    
    Ok(())
}
```

## Image Files

The `scirs2-io` module supports various image formats including PNG, JPEG, BMP, and TIFF.

### Reading Image Files

```rust
use scirs2_io::image::{read_image, read_image_metadata};

fn read_image_example() -> Result<(), Box<dyn std::error::Error>> {
    // Read an image file
    let (image_data, color_mode) = read_image("image.png")?;
    
    println!("Image information:");
    println!("  Dimensions: {} x {} pixels", image_data.shape()[0], image_data.shape()[1]);
    println!("  Color mode: {:?}", color_mode);
    println!("  Shape: {:?}", image_data.shape());
    
    // Read image metadata
    let metadata = read_image_metadata("image.png")?;
    
    println!("Image metadata:");
    println!("  Width: {} pixels", metadata.width);
    println!("  Height: {} pixels", metadata.height);
    println!("  Format: {:?}", metadata.format);
    println!("  Color mode: {:?}", metadata.color_mode);
    
    if let Some(dpi) = metadata.dpi {
        println!("  DPI: {}", dpi);
    }
    
    if !metadata.custom_metadata.is_empty() {
        println!("  Custom metadata:");
        for (key, value) in &metadata.custom_metadata {
            println!("    {}: {}", key, value);
        }
    }
    
    Ok(())
}
```

### Writing Image Files

```rust
use scirs2_io::image::{write_image, ColorMode, ImageFormat};
use ndarray::Array3;

fn write_image_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple gradient image
    let width = 256;
    let height = 256;
    
    // Create an RGB image (3 channels)
    let mut image_data = Array3::<u8>::zeros((height, width, 3));
    
    // Fill with a gradient pattern
    for y in 0..height {
        for x in 0..width {
            // Red channel - horizontal gradient
            image_data[[y, x, 0]] = x as u8;
            // Green channel - vertical gradient
            image_data[[y, x, 1]] = y as u8;
            // Blue channel - diagonal gradient
            image_data[[y, x, 2]] = ((x + y) / 2) as u8;
        }
    }
    
    // Write to different formats
    write_image("gradient.png", &image_data, ColorMode::RGB, ImageFormat::PNG)?;
    write_image("gradient.jpg", &image_data, ColorMode::RGB, ImageFormat::JPEG)?;
    write_image("gradient.bmp", &image_data, ColorMode::RGB, ImageFormat::BMP)?;
    
    println!("Images written successfully");
    
    // Create a grayscale image
    let mut grayscale_data = Array3::<u8>::zeros((height, width, 1));
    
    for y in 0..height {
        for x in 0..width {
            // Simple X+Y gradient
            grayscale_data[[y, x, 0]] = ((x + y) / 2) as u8;
        }
    }
    
    // Write grayscale image
    write_image("gradient_gray.png", &grayscale_data, ColorMode::Grayscale, ImageFormat::PNG)?;
    
    println!("Grayscale image written successfully");
    
    Ok(())
}
```

### Image Conversion and Processing

```rust
use scirs2_io::image::{read_image, write_image, convert_image, get_grayscale, ImageFormat, ColorMode};
use ndarray::Array3;

fn process_image_example() -> Result<(), Box<dyn std::error::Error>> {
    // Convert between formats
    convert_image("input.png", "output.jpg", ImageFormat::JPEG)?;
    
    println!("Image converted from PNG to JPEG");
    
    // Read an image for processing
    let (image_data, color_mode) = read_image("input.png")?;
    
    // Convert to grayscale
    let grayscale = get_grayscale(&image_data, color_mode)?;
    
    println!("Converted to grayscale: {:?}", grayscale.shape());
    
    // Save the grayscale image
    write_image("grayscale.png", &grayscale, ColorMode::Grayscale, ImageFormat::PNG)?;
    
    // Basic image processing - invert colors
    let (mut image_data, color_mode) = read_image("input.png")?;
    
    // Invert each pixel (255 - value)
    for i in 0..image_data.shape()[0] {
        for j in 0..image_data.shape()[1] {
            for k in 0..image_data.shape()[2] {
                image_data[[i, j, k]] = 255 - image_data[[i, j, k]];
            }
        }
    }
    
    // Save the inverted image
    write_image("inverted.png", &image_data, color_mode, ImageFormat::PNG)?;
    
    println!("Image processing complete");
    
    Ok(())
}
```

## JSON Serialization

JSON serialization is useful for human-readable data storage and interchange.

```rust
use scirs2_io::serialize::{serialize_array, deserialize_array, serialize_struct, deserialize_struct, SerializationFormat};
use ndarray::{Array2, ArrayD, IxDyn};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct ExperimentMetadata {
    name: String,
    author: String,
    date: String,
    parameters: Vec<f64>,
    notes: Option<String>,
}

fn json_serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D array to serialize
    let array = Array2::from_shape_vec((3, 4), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    
    // Convert to dynamic-dimensional array
    let array_dyn = array.into_dyn();
    
    // Serialize to JSON
    serialize_array("array_data.json", &array_dyn, SerializationFormat::JSON)?;
    
    println!("Array serialized to JSON successfully");
    
    // Create a struct to serialize
    let metadata = ExperimentMetadata {
        name: "Temperature Measurement".to_string(),
        author: "Researcher".to_string(),
        date: "2023-04-15".to_string(),
        parameters: vec![22.5, 0.01, 1000.0],
        notes: Some("Ambient conditions were stable during experiment".to_string()),
    };
    
    // Serialize struct to JSON
    serialize_struct("metadata.json", &metadata, SerializationFormat::JSON)?;
    
    println!("Struct serialized to JSON successfully");
    
    // Deserialize the array
    let deserialized_array: ArrayD<f64> = deserialize_array("array_data.json", SerializationFormat::JSON)?;
    
    println!("Deserialized array shape: {:?}", deserialized_array.shape());
    
    // Deserialize the struct
    let deserialized_metadata: ExperimentMetadata = deserialize_struct("metadata.json", SerializationFormat::JSON)?;
    
    println!("Deserialized metadata:");
    println!("  Name: {}", deserialized_metadata.name);
    println!("  Author: {}", deserialized_metadata.author);
    println!("  Parameters: {:?}", deserialized_metadata.parameters);
    
    Ok(())
}
```

## Binary Serialization

Binary serialization using `bincode` provides compact, efficient data storage.

```rust
use scirs2_io::serialize::{serialize_array, deserialize_array, serialize_struct, deserialize_struct, SerializationFormat};
use ndarray::{ArrayD, IxDyn};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SensorData {
    device_id: String,
    readings: Vec<f64>,
    timestamp: u64,
    valid: bool,
}

fn binary_serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create array with sensor readings
    let shape = vec![10, 5]; // 10 sensors, 5 time points
    let mut data = Vec::with_capacity(shape.iter().product());
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            data.push((i as f64) * 0.1 + (j as f64) * 0.5);
        }
    }
    
    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
    
    // Serialize to binary format
    serialize_array("sensor_readings.bin", &array, SerializationFormat::Binary)?;
    
    println!("Array serialized to binary format");
    
    // Create a collection of structs
    let sensors = vec![
        SensorData {
            device_id: "TEMP001".to_string(),
            readings: vec![20.5, 21.0, 20.8, 21.2, 20.9],
            timestamp: 1649876543,
            valid: true,
        },
        SensorData {
            device_id: "TEMP002".to_string(),
            readings: vec![22.1, 22.3, 22.0, 21.8, 22.2],
            timestamp: 1649876543,
            valid: true,
        },
        SensorData {
            device_id: "HUMID001".to_string(),
            readings: vec![45.2, 46.0, 45.5, 45.8, 46.1],
            timestamp: 1649876543,
            valid: false,
        },
    ];
    
    // Serialize struct collection
    serialize_struct("sensors.bin", &sensors, SerializationFormat::Binary)?;
    
    println!("Struct collection serialized to binary format");
    
    // Deserialize data
    let loaded_array: ArrayD<f64> = deserialize_array("sensor_readings.bin", SerializationFormat::Binary)?;
    
    println!("Deserialized array shape: {:?}", loaded_array.shape());
    
    let loaded_sensors: Vec<SensorData> = deserialize_struct("sensors.bin", SerializationFormat::Binary)?;
    
    println!("Deserialized {} sensor records", loaded_sensors.len());
    
    // Verify data integrity
    assert_eq!(sensors, loaded_sensors);
    println!("Binary serialization verified successfully");
    
    Ok(())
}
```

## MessagePack Serialization

MessagePack provides compact, cross-language binary serialization.

```rust
use scirs2_io::serialize::{serialize_array, deserialize_array, serialize_struct, deserialize_struct, SerializationFormat};
use ndarray::{Array, IxDyn};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
struct DataPoint {
    x: f64,
    y: f64,
    label: String,
    properties: HashMap<String, String>,
}

fn messagepack_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3D array
    let shape = vec![4, 4, 4];
    let array = Array::from_shape_fn(IxDyn(&shape), |idx| {
        (idx[0] + idx[1] + idx[2]) as f64
    });
    
    // Serialize to MessagePack
    serialize_array("array3d.msgpack", &array, SerializationFormat::MessagePack)?;
    
    println!("3D array serialized to MessagePack");
    
    // Create a collection of data points
    let mut points = Vec::new();
    
    for i in 0..10 {
        let mut properties = HashMap::new();
        properties.insert("group".to_string(), format!("Cluster {}", i % 3 + 1));
        properties.insert("quality".to_string(), if i % 2 == 0 { "high" } else { "medium" }.to_string());
        
        points.push(DataPoint {
            x: i as f64 * 0.5,
            y: (i as f64 * 0.5).sin(),
            label: format!("Point {}", i + 1),
            properties,
        });
    }
    
    // Serialize to MessagePack
    serialize_struct("points.msgpack", &points, SerializationFormat::MessagePack)?;
    
    println!("Data points serialized to MessagePack");
    
    // Deserialize the data
    let loaded_array: Array<f64, IxDyn> = deserialize_array("array3d.msgpack", SerializationFormat::MessagePack)?;
    
    println!("Deserialized array shape: {:?}", loaded_array.shape());
    
    let loaded_points: Vec<DataPoint> = deserialize_struct("points.msgpack", SerializationFormat::MessagePack)?;
    
    println!("Deserialized {} data points", loaded_points.len());
    
    // Display some data
    if !loaded_points.is_empty() {
        let sample = &loaded_points[0];
        println!("Sample point: ({}, {}), Label: {}", sample.x, sample.y, sample.label);
        
        if let Some(group) = sample.properties.get("group") {
            println!("  Group: {}", group);
        }
    }
    
    Ok(())
}
```

## Combining Multiple File Formats

In real-world scientific applications, you often need to work with multiple file formats together. Here's an example that combines several formats:

```rust
use scirs2_io::csv::{read_csv, CsvReaderConfig};
use scirs2_io::image::{write_image, ColorMode, ImageFormat};
use scirs2_io::serialize::{serialize_struct, SerializationFormat};
use scirs2_io::matlab::MatFile;
use scirs2_io::compression::{compress_file, CompressionAlgorithm};
use scirs2_io::validation::{calculate_file_checksum, ChecksumAlgorithm};
use ndarray::{Array2, Array3};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct AnalysisResults {
    dataset: String,
    timestamp: u64,
    statistics: HashMap<String, f64>,
    visualization_files: Vec<String>,
    checksums: HashMap<String, String>,
}

fn multi_format_example() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read data from CSV
    let config = CsvReaderConfig {
        delimiter: ',',
        has_header: true,
        ..Default::default()
    };
    
    let (headers, data) = read_csv("measurements.csv", Some(config))?;
    
    println!("Loaded data with {} rows, {} columns", data.nrows(), data.ncols());
    
    // 2. Extract numeric columns for analysis
    let numeric_data = Array2::<f64>::from_shape_fn((data.nrows(), data.ncols()), |(i, j)| {
        data[[i, j]].parse::<f64>().unwrap_or(0.0)
    });
    
    // 3. Perform statistical analysis
    let mut statistics = HashMap::new();
    
    for (j, header) in headers.iter().enumerate() {
        if j < numeric_data.ncols() {
            let column = numeric_data.column(j);
            
            // Calculate mean
            let mean = column.sum() / column.len() as f64;
            statistics.insert(format!("{}_mean", header), mean);
            
            // Calculate standard deviation
            let variance = column.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / column.len() as f64;
            let std_dev = variance.sqrt();
            statistics.insert(format!("{}_std", header), std_dev);
        }
    }
    
    // 4. Generate visualization (a heatmap image)
    let mut heatmap = Array3::<u8>::zeros((data.nrows(), data.ncols(), 3));
    
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            let value = numeric_data[[i, j]];
            // Normalize to 0-255 range (assuming values between 0-100)
            let normalized = ((value / 100.0) * 255.0).min(255.0).max(0.0) as u8;
            
            // Use a simple color gradient (blue to red)
            heatmap[[i, j, 0]] = normalized;          // Red
            heatmap[[i, j, 1]] = 0;                   // Green
            heatmap[[i, j, 2]] = 255 - normalized;    // Blue
        }
    }
    
    // Save the heatmap
    write_image("heatmap.png", &heatmap, ColorMode::RGB, ImageFormat::PNG)?;
    
    // 5. Save results to MATLAB for further analysis
    let mut mat_file = MatFile::new();
    mat_file.add_array("data", &numeric_data)?;
    
    // Add statistics as scalars
    for (name, value) in &statistics {
        mat_file.add_scalar(name, *value)?;
    }
    
    mat_file.save("analysis.mat")?;
    
    // 6. Collect all results and serialize to JSON
    let results = AnalysisResults {
        dataset: "measurements.csv".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        statistics,
        visualization_files: vec!["heatmap.png".to_string(), "analysis.mat".to_string()],
        checksums: HashMap::new(),
    };
    
    serialize_struct("results.json", &results, SerializationFormat::JSON)?;
    
    // 7. Generate checksums and compress all output files
    let files = ["heatmap.png", "analysis.mat", "results.json"];
    
    for file in &files {
        // Generate checksum
        let checksum = calculate_file_checksum(file, ChecksumAlgorithm::SHA256)?;
        println!("Checksum for {}: {}", file, checksum);
        
        // Compress the file
        let compressed_path = compress_file(file, None, CompressionAlgorithm::Zstd, Some(6))?;
        println!("Compressed {} to {}", file, compressed_path);
    }
    
    println!("Multi-format data processing complete");
    
    Ok(())
}
```

## Performance Considerations

When working with different file formats, consider these performance tips:

1. **CSV files**:
   - Use chunked reading for large files
   - Consider column-oriented operations
   - For numeric-only data, use numeric conversion

2. **Image files**:
   - JPEG offers good compression but is lossy
   - PNG provides lossless compression but larger files
   - Use appropriate color modes (Grayscale vs. RGB)

3. **Serialization**:
   - Binary formats (oxicode, MessagePack) are faster and more compact
   - JSON is human-readable but larger and slower
   - For scientific data, consider compression

4. **MATLAB files**:
   - Efficient for matrix storage
   - Good compatibility with MATLAB software
   - Limited support for complex types

5. **WAV files**:
   - Uncompressed format (large files)
   - Fast reading/writing
   - Consider compression for storage

## Best Practices

1. **Choose the right format for your data**:
   - Tabular data: CSV
   - Images: PNG (lossless) or JPEG (lossy)
   - Scientific data: MATLAB, HDF5, or custom serialization
   - Configuration: JSON

2. **Always validate data integrity**:
   - Use checksums for verification
   - Validate format-specific constraints
   - Check numeric ranges

3. **Consider compression**:
   - Balance between speed and compression ratio
   - Use ZSTD for general purpose compression
   - Use LZ4 for speed-critical applications

4. **Handle errors gracefully**:
   - Check for file existence before operations
   - Validate format before processing
   - Provide meaningful error messages

By following these examples and best practices, you can effectively work with various file formats in your scientific applications using the `scirs2-io` module.