//! Basic usage example for OxiCode

use oxicode::{config, Decode, Encode};

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
struct Person {
    name: String,
    age: u32,
    email: String,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
enum Message {
    Text(String),
    Number(u64),
    Pair(String, String),
}

fn main() -> Result<(), oxicode::Error> {
    println!("OxiCode Basic Usage Example\n");

    // Example 1: Simple struct encoding
    println!("1. Encoding a struct:");
    let person = Person {
        name: "Alice".to_string(),
        age: 30,
        email: "alice@example.com".to_string(),
    };

    let bytes = oxicode::encode_to_vec(&person)?;
    println!("   Encoded {} bytes", bytes.len());

    let (decoded, bytes_read): (Person, _) = oxicode::decode_from_slice(&bytes)?;
    println!("   Decoded {} bytes", bytes_read);
    assert_eq!(person, decoded);
    println!("   ✓ Round-trip successful\n");

    // Example 2: Enum encoding
    println!("2. Encoding an enum:");
    let msg = Message::Pair("key".to_string(), "value".to_string());

    let bytes = oxicode::encode_to_vec(&msg)?;
    println!("   Encoded {} bytes", bytes.len());

    let (decoded, _): (Message, _) = oxicode::decode_from_slice(&bytes)?;
    assert_eq!(msg, decoded);
    println!("   ✓ Enum round-trip successful\n");

    // Example 3: Using different configurations
    println!("3. Configuration comparison:");

    let data = vec![1u32, 2, 3, 4, 5];

    // Standard config (varint encoding)
    let standard_bytes = oxicode::encode_to_vec_with_config(&data, config::standard())?;
    println!("   Standard config: {} bytes", standard_bytes.len());

    // Legacy config (fixed-int encoding, bincode 1.0 compatible)
    let legacy_bytes = oxicode::encode_to_vec_with_config(&data, config::legacy())?;
    println!("   Legacy config:   {} bytes", legacy_bytes.len());

    println!("   ✓ Varint encoding is more compact for small values\n");

    // Example 4: Collections
    println!("4. Encoding collections:");
    let vec_data = vec![10, 20, 30, 40, 50];
    let bytes = oxicode::encode_to_vec(&vec_data)?;
    let (decoded, _): (Vec<i32>, _) = oxicode::decode_from_slice(&bytes)?;
    assert_eq!(vec_data, decoded);
    println!("   ✓ Vec<i32> round-trip successful\n");

    // Example 5: Nested structures
    println!("5. Encoding nested structures:");
    let nested = vec![Some(person.clone()), None, Some(person)];
    let bytes = oxicode::encode_to_vec(&nested)?;
    let (decoded, _): (Vec<Option<Person>>, _) = oxicode::decode_from_slice(&bytes)?;
    assert_eq!(nested, decoded);
    println!("   ✓ Vec<Option<Person>> round-trip successful\n");

    println!("All examples completed successfully!");

    Ok(())
}
