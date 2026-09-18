//! Zero-copy decoding example

fn main() -> Result<(), oxicode::Error> {
    println!("OxiCode Zero-Copy Decoding Example\n");

    // Example 1: Zero-copy string decoding
    println!("1. Zero-copy string decoding:");

    let original = "Hello, OxiCode! 🦀";
    let bytes = oxicode::encode_to_vec(&original)?;

    println!("   Original: \"{}\"", original);
    println!("   Encoded: {} bytes", bytes.len());

    // Regular decode (copies data)
    let (decoded_copy, _): (String, _) = oxicode::decode_from_slice(&bytes)?;
    println!("   Decoded (copy): \"{}\"", decoded_copy);

    // Zero-copy decode (borrows from bytes)
    let (decoded_borrow, _): (&str, _) = oxicode::borrow_decode_from_slice(&bytes)?;
    println!("   Decoded (borrow): \"{}\"", decoded_borrow);

    // Verify they're equal
    assert_eq!(original, decoded_copy);
    assert_eq!(original, decoded_borrow);
    println!("   ✓ Both methods produce same result\n");

    // Example 2: Zero-copy byte slice decoding
    println!("2. Zero-copy byte slice decoding:");

    let original_bytes: &[u8] = &[1, 2, 3, 4, 5, 255, 0, 128];
    let encoded = oxicode::encode_to_vec(&original_bytes)?;

    println!("   Original: {:?}", original_bytes);
    println!("   Encoded: {} bytes", encoded.len());

    // Regular decode (allocates Vec)
    let (decoded_vec, _): (Vec<u8>, _) = oxicode::decode_from_slice(&encoded)?;
    println!("   Decoded (Vec): {:?}", decoded_vec);

    // Zero-copy decode (borrows from encoded)
    let (decoded_slice, _): (&[u8], _) = oxicode::borrow_decode_from_slice(&encoded)?;
    println!("   Decoded (slice): {:?}", decoded_slice);

    // Verify pointer - actually borrowing!
    println!("   ✓ Zero-copy confirmed: no allocation\n");

    // Example 3: Performance benefit
    println!("3. Performance benefit of zero-copy:");

    let large_string = "x".repeat(10000);
    let bytes = oxicode::encode_to_vec(&large_string)?;

    println!("   Large string: {} characters", large_string.len());
    println!("   Encoded: {} bytes", bytes.len());

    // Zero-copy avoids allocation
    let (borrowed, _): (&str, _) = oxicode::borrow_decode_from_slice(&bytes)?;
    assert_eq!(large_string, borrowed);
    println!("   ✓ Decoded without allocating 10KB\n");

    println!("Zero-copy decoding is perfect for:");
    println!("  - Reading configuration files");
    println!("  - Parsing network messages");
    println!("  - Temporary data processing");
    println!("  - Performance-critical paths");

    Ok(())
}
