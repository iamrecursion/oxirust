use scirs2_io::validation::{
    calculate_checksum, calculate_file_checksum, create_checksum_file, create_directory_manifest,
    generate_file_integrity_metadata, generate_validation_report, load_integrity_metadata,
    save_integrity_metadata, validate_file_integrity, verify_checksum, verify_checksum_file,
    verify_file_checksum, ChecksumAlgorithm,
};

use scirs2_io::validation::formats::{
    detect_file_format, get_scientific_format_validators, validate_file_format, validate_format,
    DataFormat,
};

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary files for examples
    println!("\n=== Creating Example Files ===");
    let temp_dir = tempfile::tempdir()?;
    let test_filepath = temp_dir.path().join("testdata.txt");
    let test_csv_path = temp_dir.path().join("testdata.csv");
    let test_json_path = temp_dir.path().join("testdata.json");

    // Create a test file
    let testdata = "This is test data for demonstrating data validation and integrity checking.";
    std::fs::write(&test_filepath, testdata)?;
    println!("Created test file: {}", test_filepath.display());

    // Create a test CSV file
    let csvdata = "id,name,value\n1,item1,10.5\n2,item2,20.3\n3,item3,15.8";
    std::fs::write(&test_csv_path, csvdata)?;
    println!("Created test CSV file: {}", test_csv_path.display());

    // Create a test JSON file
    let jsondata = r#"{"data": [{"id": 1, "value": 10.5}, {"id": 2, "value": 20.3}]}"#;
    std::fs::write(&test_json_path, jsondata)?;
    println!("Created test JSON file: {}", test_json_path.display());

    // Basic checksum operations
    basic_checksum_example(testdata)?;

    // File checksum operations
    file_checksum_example(&test_filepath)?;

    // Integrity metadata example
    integrity_metadata_example(&test_filepath, temp_dir.path())?;

    // Format validation example
    format_validation_example(&test_csv_path, &test_json_path)?;

    // Directory manifest example
    directory_manifest_example(temp_dir.path())?;

    println!("\nAll validation examples completed successfully!");
    Ok(())
}

#[allow(dead_code)]
fn basic_checksum_example(data: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Basic Checksum Example ===");

    // Generate checksums with different algorithms
    let crc32_checksum = calculate_checksum(data.as_bytes(), ChecksumAlgorithm::CRC32);
    let sha256_checksum = calculate_checksum(data.as_bytes(), ChecksumAlgorithm::SHA256);
    let blake3_checksum = calculate_checksum(data.as_bytes(), ChecksumAlgorithm::BLAKE3);

    println!("Original data: \"{}\"", data);
    println!("CRC32 checksum: {}", crc32_checksum);
    println!("SHA256 checksum: {}", sha256_checksum);
    println!("BLAKE3 checksum: {}", blake3_checksum);

    // Verify checksums
    println!("\nVerifying checksums:");
    println!(
        "CRC32 valid: {}",
        verify_checksum(data.as_bytes(), &crc32_checksum, ChecksumAlgorithm::CRC32)
    );
    println!(
        "SHA256 valid: {}",
        verify_checksum(data.as_bytes(), &sha256_checksum, ChecksumAlgorithm::SHA256)
    );
    println!(
        "BLAKE3 valid: {}",
        verify_checksum(data.as_bytes(), &blake3_checksum, ChecksumAlgorithm::BLAKE3)
    );

    // Verify with modified data
    let modifieddata = format!("{}!", data);
    println!("\nVerifying checksums with modified data:");
    println!(
        "CRC32 valid: {}",
        verify_checksum(
            modifieddata.as_bytes(),
            &crc32_checksum,
            ChecksumAlgorithm::CRC32
        )
    );
    println!(
        "SHA256 valid: {}",
        verify_checksum(
            modifieddata.as_bytes(),
            &sha256_checksum,
            ChecksumAlgorithm::SHA256
        )
    );
    println!(
        "BLAKE3 valid: {}",
        verify_checksum(
            modifieddata.as_bytes(),
            &blake3_checksum,
            ChecksumAlgorithm::BLAKE3
        )
    );

    Ok(())
}

#[allow(dead_code)]
fn file_checksum_example(filepath: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== File Checksum Example ===");

    // Calculate checksums for the file
    println!("Calculating file checksums for: {}", filepath.display());

    let crc32_checksum = calculate_file_checksum(filepath, ChecksumAlgorithm::CRC32)?;
    let sha256_checksum = calculate_file_checksum(filepath, ChecksumAlgorithm::SHA256)?;

    println!("CRC32 checksum: {}", crc32_checksum);
    println!("SHA256 checksum: {}", sha256_checksum);

    // Verify file checksums
    println!("\nVerifying file checksums:");
    println!(
        "CRC32 valid: {}",
        verify_file_checksum(filepath, &crc32_checksum, ChecksumAlgorithm::CRC32)?
    );
    println!(
        "SHA256 valid: {}",
        verify_file_checksum(filepath, &sha256_checksum, ChecksumAlgorithm::SHA256)?
    );

    // Create a checksum file
    let checksum_filepath =
        create_checksum_file(filepath, ChecksumAlgorithm::SHA256, None::<String>)?;
    println!("\nCreated checksum file: {}", checksum_filepath);

    // Verify using the checksum file
    let verification_result = verify_checksum_file(filepath, &checksum_filepath)?;
    println!("Verification using checksum file: {}", verification_result);

    Ok(())
}

#[allow(dead_code)]
fn integrity_metadata_example(
    filepath: &Path,
    temp_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Integrity Metadata Example ===");

    // Generate integrity metadata
    let metadata = generate_file_integrity_metadata(filepath, ChecksumAlgorithm::SHA256)?;

    println!("Generated integrity metadata:");
    println!("  File size: {} bytes", metadata.size);
    println!("  Checksum algorithm: {}", metadata.algorithm);
    println!("  Checksum: {}", metadata.checksum);
    println!("  Timestamp: {}", metadata.timestamp);

    // Save metadata to a file
    let metadata_path = temp_dir.join("integrity_metadata.json");
    save_integrity_metadata(&metadata, &metadata_path)?;
    println!("\nSaved integrity metadata to: {}", metadata_path.display());

    // Load metadata from file
    let loaded_metadata = load_integrity_metadata(&metadata_path)?;
    println!("Loaded metadata checksum: {}", loaded_metadata.checksum);

    // Validate file against metadata
    let validation_result = validate_file_integrity(filepath, &metadata)?;
    println!("File validation result: {}", validation_result);

    // Generate validation report
    let report = generate_validation_report(filepath, &metadata)?;
    println!("\nValidation Report:");
    println!("{}", report.format());

    // Create a modified file to demonstrate validation failure
    let modified_filepath = temp_dir.join("modified_testdata.txt");
    let original_content = fs::read_to_string(filepath)?;
    let modified_content = format!("{}!", original_content); // Append an exclamation mark
    fs::write(&modified_filepath, modified_content)?;

    // Validate the modified file against the original metadata
    let modified_validation = validate_file_integrity(&modified_filepath, &metadata)?;
    println!("\nModified file validation result: {}", modified_validation);

    let modified_report = generate_validation_report(&modified_filepath, &metadata)?;
    println!("Modified File Validation Report:");
    println!("{}", modified_report.format());

    Ok(())
}

#[allow(dead_code)]
fn format_validation_example(
    csv_path: &Path,
    json_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Format Validation Example ===");

    // Get format validators
    let _validators = get_scientific_format_validators();

    // Detect formats
    println!("Detecting file formats:");

    let csv_format = detect_file_format(csv_path)?;
    println!(
        "Detected format for {}: {:?}",
        csv_path.display(),
        csv_format
    );

    let json_format = detect_file_format(json_path)?;
    println!(
        "Detected format for {}: {:?}",
        json_path.display(),
        json_format
    );

    // Validate specific formats
    println!("\nValidating specific formats:");

    let csv_validation = validate_format(csv_path, DataFormat::CSV)?;
    println!(
        "CSV validation for {}: {}",
        csv_path.display(),
        csv_validation
    );

    let json_validation = validate_format(json_path, DataFormat::JSON)?;
    println!(
        "JSON validation for {}: {}",
        json_path.display(),
        json_validation
    );

    // Create an invalid CSV file
    let invalid_csv_path = csv_path.with_file_name("invalid.csv");
    let invalid_csv_content = "header1,header2,header3\nvalue1,value2\nvalue3,value4,value5,value6";
    fs::write(&invalid_csv_path, invalid_csv_content)?;

    // Detailed format validation
    println!("\nDetailed format validation:");

    let csv_detailed = validate_file_format(csv_path, DataFormat::CSV)?;
    println!(
        "CSV detailed validation: valid={}, details={:?}",
        csv_detailed.valid, csv_detailed.details
    );

    let invalid_csv_detailed = validate_file_format(&invalid_csv_path, DataFormat::CSV)?;
    println!(
        "Invalid CSV detailed validation: valid={}, details={:?}",
        invalid_csv_detailed.valid, invalid_csv_detailed.details
    );

    let json_detailed = validate_file_format(json_path, DataFormat::JSON)?;
    println!(
        "JSON detailed validation: valid={}, details={:?}",
        json_detailed.valid, json_detailed.details
    );

    Ok(())
}

#[allow(dead_code)]
fn directory_manifest_example(dirpath: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Directory Manifest Example ===");

    // Create directory manifest
    let manifest_path = dirpath.join("manifest.json");
    create_directory_manifest(dirpath, &manifest_path, ChecksumAlgorithm::SHA256, true)?;

    println!("Created directory manifest: {}", manifest_path.display());

    // Load the manifest and verify
    let manifest_content = fs::read_to_string(&manifest_path)?;
    let manifest: scirs2_io::validation::DirectoryManifest =
        serde_json::from_str(&manifest_content)?;

    println!("Manifest contains {} files", manifest.files.len());

    // Verify the directory against the manifest
    let verification = manifest.verify_directory(dirpath)?;
    println!("Directory verification result: {}", verification.passed());
    println!("Verified {} files", verification.verified_files.len());

    // Create a new file to demonstrate manifest verification failure
    let new_filepath = dirpath.join("new_file.txt");
    let mut new_file = File::create(&new_filepath)?;
    writeln!(new_file, "This file was created after the manifest")?;

    // Modify a file in the directory
    let first_file = if !manifest.files.is_empty() {
        let first_path = &manifest.files[0].path;
        let full_path = dirpath.join(first_path);

        if full_path.exists() {
            let mut content = fs::read_to_string(&full_path)?;
            content.push_str("\nModified content");
            fs::write(&full_path, content)?;
            Some(first_path.clone())
        } else {
            None
        }
    } else {
        None
    };

    // Verify again
    let verification_after = manifest.verify_directory(dirpath)?;
    println!("\nVerification after changes:");
    println!("Passed: {}", verification_after.passed());
    println!(
        "Verified files: {}",
        verification_after.verified_files.len()
    );
    println!("Missing files: {}", verification_after.missing_files.len());
    println!(
        "Modified files: {}",
        verification_after.modified_files.len()
    );

    if let Some(path) = first_file {
        if verification_after.modified_files.contains(&path) {
            println!("Successfully detected modification to: {}", path);
        }
    }

    // Save the verification report
    let report_path = dirpath.join("verification_report.json");
    verification_after.save(&report_path)?;
    println!("Saved verification report to: {}", report_path.display());

    Ok(())
}
