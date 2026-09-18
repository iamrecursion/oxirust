//! Example demonstrating the convenient array! macro usage in scirs2-core
//!
//! This example shows how users can now use the array! macro directly from scirs2_core
//! instead of having to import it from scirs2_autograd::ndarray

// Before this fix, users had to do:
// use scirs2_autograd::ndarray::array;

// Now they can simply do:
use scirs2_core::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Testing array! macro re-export from scirs2_core");

    // Create a 1D array
    let vector = array![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("1D array: {:?}", vector);
    println!("Shape: {:?}", vector.shape());

    // Create a 2D matrix
    let matrix = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    println!("2D matrix: {:?}", matrix);
    println!("Shape: {:?}", matrix.shape());

    // Create a 3D tensor
    let tensor = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    println!("3D tensor: {:?}", tensor);
    println!("Shape: {:?}", tensor.shape());

    // Test accessing elements
    println!("matrix[1, 2] = {}", matrix[[1, 2]]);
    println!("tensor[1, 0, 1] = {}", tensor[[1, 0, 1]]);

    println!("✅ Array macro re-export working correctly!");

    Ok(())
}
