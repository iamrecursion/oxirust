//! Configuration and advanced features example

use oxicode::{config, Decode, Encode};

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
struct Data {
    id: u64,
    values: Vec<i32>,
}

fn main() -> Result<(), oxicode::Error> {
    println!("OxiCode Configuration Examples\n");

    let data = Data {
        id: 12345,
        values: vec![1, 2, 3, -100, 1000],
    };

    // Example 1: Standard configuration (little-endian + varint)
    println!("1. Standard configuration:");
    let cfg = config::standard();
    let bytes = oxicode::encode_to_vec_with_config(&data, cfg)?;
    println!("   Size: {} bytes (varint encoding)", bytes.len());
    let (decoded, _): (Data, _) = oxicode::decode_from_slice_with_config(&bytes, cfg)?;
    assert_eq!(data, decoded);
    println!("   ✓ Verified\n");

    // Example 2: Legacy configuration (bincode 1.0 compatible)
    println!("2. Legacy configuration (bincode 1.0 compatible):");
    let cfg = config::legacy();
    let bytes = oxicode::encode_to_vec_with_config(&data, cfg)?;
    println!("   Size: {} bytes (fixed-int encoding)", bytes.len());
    let (decoded, _): (Data, _) = oxicode::decode_from_slice_with_config(&bytes, cfg)?;
    assert_eq!(data, decoded);
    println!("   ✓ Compatible with bincode 1.0\n");

    // Example 3: Big-endian encoding
    println!("3. Big-endian encoding:");
    let cfg = config::standard().with_big_endian();
    let bytes = oxicode::encode_to_vec_with_config(&data, cfg)?;
    println!("   Size: {} bytes", bytes.len());
    let (decoded, _): (Data, _) = oxicode::decode_from_slice_with_config(&bytes, cfg)?;
    assert_eq!(data, decoded);
    println!("   ✓ Big-endian verified\n");

    // Example 4: Fixed-int encoding
    println!("4. Fixed-int encoding:");
    let cfg = config::standard().with_fixed_int_encoding();
    let bytes = oxicode::encode_to_vec_with_config(&data, cfg)?;
    println!("   Size: {} bytes (no varint compression)", bytes.len());
    let (decoded, _): (Data, _) = oxicode::decode_from_slice_with_config(&bytes, cfg)?;
    assert_eq!(data, decoded);
    println!("   ✓ Fixed-int verified\n");

    // Example 5: Memory limit
    println!("5. Memory limit configuration:");
    let cfg = config::standard().with_limit::<1024>();
    let small_data = Data {
        id: 1,
        values: vec![1, 2, 3],
    };
    let bytes = oxicode::encode_to_vec_with_config(&small_data, cfg)?;
    println!("   Small data: {} bytes (within 1KB limit)", bytes.len());
    let (decoded, _): (Data, _) = oxicode::decode_from_slice_with_config(&bytes, cfg)?;
    assert_eq!(small_data, decoded);
    println!("   ✓ Within limit\n");

    // Example 6: Encoding to fixed buffer
    println!("6. Encoding to fixed buffer:");
    let mut buffer = [0u8; 256];
    let cfg = config::standard();
    let bytes_written = oxicode::encode_into_slice(data.clone(), &mut buffer, cfg)?;
    println!("   Wrote {} bytes to buffer", bytes_written);
    let (decoded, _): (Data, _) = oxicode::decode_from_slice(&buffer[..bytes_written])?;
    assert_eq!(data, decoded);
    println!("   ✓ Fixed buffer encoding works\n");

    // Example 7: Comparison of configurations
    println!("7. Configuration size comparison:");
    let test_value = 1000u64;

    let standard = oxicode::encode_to_vec_with_config(&test_value, config::standard())?;
    let legacy = oxicode::encode_to_vec_with_config(&test_value, config::legacy())?;
    let big_endian =
        oxicode::encode_to_vec_with_config(&test_value, config::standard().with_big_endian())?;

    println!("   Value: {}", test_value);
    println!(
        "   Standard (varint, little-endian): {} bytes",
        standard.len()
    );
    println!(
        "   Legacy (fixed, little-endian):    {} bytes",
        legacy.len()
    );
    println!(
        "   Big-endian (varint, big-endian):  {} bytes",
        big_endian.len()
    );
    println!("   ✓ Varint saves space for small values\n");

    println!("All configuration examples completed successfully!");

    Ok(())
}
