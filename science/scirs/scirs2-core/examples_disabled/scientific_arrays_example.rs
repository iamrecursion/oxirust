use ndarray::{Array, Array1, Array2, Axis};
use scirs2_core::array::{
    mask_array, masked_equal, masked_invalid, ArrayError, FieldValue,
    MaskedArray, Record, RecordArray,
};
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> Result<(), ArrayError> {
    println!("Scientific Arrays Example");
    println!("========================\n");

    // Example 1: Masked Arrays
    println!("Example 1: Masked Arrays");
    println!("---------------------");

    // Create a sample array with some invalid values
    let data = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, f64::INFINITY, 7.0]);
    println!("Original array: {:?}", data);

    // Create a masked array that masks out invalid values
    let masked = masked_invalid(&data);
    println!("Masked array (masking invalid values): {:?}", masked);

    // Create a masked array that masks specific values
    let masked_twos = masked_equal(&data, 2.0);
    println!("Masked array (masking 2.0): {:?}", masked_twos);

    // Create a custom masked array with a specific mask
    let mask = Array1::from_vec(vec![false, false, true, false, true, false, false]);
    let custom_masked = mask_array(data.clone(), Some(mask), Some(0.0))?;
    println!("Custom masked array: {:?}", custom_masked);

    // Operations with masked arrays
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let mask_a = Array1::from_vec(vec![false, true, false, false, false]);
    let mask_b = Array1::from_vec(vec![false, false, false, true, false]);

    let masked_a = mask_array(a.clone(), Some(mask_a), Some(0.0))?;
    let masked_b = mask_array(b.clone(), Some(mask_b), Some(0.0))?;

    println!("\nOperations with masked arrays:");
    println!("maskeda: {:?}", masked_a);
    println!("maskedb: {:?}", masked_b);

    // Addition
    let sum = &masked_a + &masked_b;
    println!("masked_a + masked_b = {:?}", sum);

    // Subtraction
    let diff = &masked_a - &masked_b;
    println!("masked_a - masked_b = {:?}", diff);

    // Multiplication
    let prod = &masked_a * &masked_b;
    println!("masked_a * masked_b = {:?}", prod);

    // Division
    let div = &masked_a / &masked_b;
    println!("masked_a / masked_b = {:?}", div);

    // Example 2: Record Arrays
    println!("\nExample 2: Record Arrays");
    println!("---------------------");

    // Create field values for different types
    let names = vec![
        FieldValue::String("Alice".to_string()),
        FieldValue::String("Bob".to_string()),
        FieldValue::String("Charlie".to_string()),
    ];

    let ages = vec![
        FieldValue::Int(30),
        FieldValue::Int(25),
        FieldValue::Int(35),
    ];

    let heights = vec![
        FieldValue::Float(1.75),
        FieldValue::Float(1.85),
        FieldValue::Float(1.70),
    ];

    let is_active = vec![
        FieldValue::Bool(true),
        FieldValue::Bool(false),
        FieldValue::Bool(true),
    ];

    // Create records by combining the field values
    let mut records = Vec::new();
    for i in 0..3 {
        let mut record = Record::new();
        record.set_field("name", names[i].clone());
        record.set_field("age", ages[i].clone());
        record.set_field("height", heights[i].clone());
        record.set_field("active", is_active[i].clone());
        records.push(record);
    }

    let record_array = RecordArray::new(records)?;
    println!("Record array: {:?}", record_array);

    // Access records
    println!("\nAccessing records:");
    for i in 0..record_array.len() {
        let record = record_array.get_record(i)?;
        println!("Record {}: {:?}", i, record);

        // Access fields directly
        let name = record.get_field_as_string(name)?;
        let age = record.get_field_as_int(age)?;
        let height = record.get_field_as_float(height)?;
        let active = record.get_field_as_bool(active)?;

        println!(
            "  Name: {}, Age: {}, Height: {}, Active: {}",
            name, age, height, active
        );
    }

    // Access fields directly from the record array
    println!("\nAccessing fields directly:");
    let all_names = record_array.get_field_as_string_array(name)?;
    let all_ages = record_array.get_field_as_int_array(age)?;

    println!("All names: {:?}", all_names);
    println!("All ages: {:?}", all_ages);

    // Filter records
    println!("\nFiltering records:");
    let active_records =
        record_array.filter(|rec| rec.get_field_as_bool(active).unwrap_or(false))?;

    println!("Active records: {:?}", active_records);

    Ok(())
}
