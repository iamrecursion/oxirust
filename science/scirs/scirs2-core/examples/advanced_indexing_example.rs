// Example demonstrating advanced indexing and statistical functions

use scirs2_core::ndarray_ext::{array, Array2, Axis};
use scirs2_core::ndarray_ext::{indexing, stats};

#[allow(dead_code)]
fn main() {
    println!("SciRS2-Core Advanced Indexing and Statistics Example");
    println!("===================================================\n");

    // Create some example arrays
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let mask = array![
        [true, false, true],
        [false, true, false],
        [true, false, true]
    ];

    println!("Original array:");
    println!("{a:?}\n");

    // 1. Advanced Indexing Examples
    println!("Advanced Indexing Examples:");
    println!("-------------------------");

    // Boolean masking
    let selected = indexing::boolean_mask_2d(a.view(), mask.view()).expect("Operation failed");
    println!("Boolean masking:");
    println!("Mask: {mask:?}");
    println!("Selected elements: {selected:?}\n");

    // Fancy indexing with explicit indices
    let row_indices = array![0, 2];
    let col_indices = array![0, 1];
    let fancy_indexed = indexing::fancy_index_2d(a.view(), row_indices.view(), col_indices.view())
        .expect("Operation failed");
    println!("Fancy indexing with row_indices=[0, 2], col_indices=[0, 1]:");
    println!("Selected elements: {fancy_indexed:?}\n");

    // Extract a diagonal
    let main_diag = indexing::diagonal(a.view(), 0).expect("Operation failed");
    println!("Main diagonal:");
    println!("{main_diag:?}\n");

    // Extract elements matching a condition
    let condition_result = indexing::where_2d(a.view(), |&x| x > 5.0).expect("Operation failed");
    println!("Elements > 5.0:");
    println!("{condition_result:?}\n");

    // 2. Statistical Function Examples
    println!("Statistical Function Examples:");
    println!("----------------------------");

    // Basic statistics
    println!("Mean:");
    println!(
        "- Global: {:?}",
        stats::mean(&a.view(), None).expect("Operation failed")
    );
    println!(
        "- By column: {:?}",
        stats::mean(&a.view(), Some(Axis(0))).expect("Operation failed")
    );
    println!(
        "- By row: {:?}\n",
        stats::mean(&a.view(), Some(Axis(1))).expect("Operation failed")
    );

    println!("Median:");
    println!(
        "- Global: {:?}",
        stats::median(&a.view(), None).expect("Operation failed")
    );
    println!(
        "- By column: {:?}",
        stats::median(&a.view(), Some(Axis(0))).expect("Operation failed")
    );
    println!(
        "- By row: {:?}\n",
        stats::median(&a.view(), Some(Axis(1))).expect("Operation failed")
    );

    println!("Standard Deviation (sample):");
    println!(
        "- Global: {:?}",
        stats::std_dev(&a.view(), None, 1).expect("Operation failed")
    );
    println!(
        "- By column: {:?}",
        stats::std_dev(&a.view(), Some(Axis(0)), 1).expect("Operation failed")
    );
    println!(
        "- By row: {:?}\n",
        stats::std_dev(&a.view(), Some(Axis(1)), 1).expect("Operation failed")
    );

    println!("Min/Max:");
    println!(
        "- Min: {:?}",
        stats::min(&a.view(), None).expect("Operation failed")
    );
    println!(
        "- Max: {:?}",
        stats::max(&a.view(), None).expect("Operation failed")
    );
    println!(
        "- Min by column: {:?}",
        stats::min(&a.view(), Some(Axis(0))).expect("Operation failed")
    );
    println!(
        "- Max by row: {:?}\n",
        stats::max(&a.view(), Some(Axis(1))).expect("Operation failed")
    );

    println!("Percentiles:");
    println!(
        "- 25th percentile: {:?}",
        stats::percentile(&a.view(), 25.0, None).expect("Operation failed")
    );
    println!(
        "- 50th percentile: {:?}",
        stats::percentile(&a.view(), 50.0, None).expect("Operation failed")
    );
    println!(
        "- 75th percentile: {:?}",
        stats::percentile(&a.view(), 75.0, None).expect("Operation failed")
    );
    println!(
        "- 75th percentile by column: {:?}\n",
        stats::percentile(&a.view(), 75.0, Some(Axis(0))).expect("Operation failed")
    );

    // 3. Transformation with statistics example
    println!("Transformation Example: Standardization (Z-scores)");
    println!("---------------------------------------------");

    // Calculate the z-scores (standardize the data)
    let z_scores = standardize_array(&a);
    println!("Z-scores:");
    println!("{z_scores:?}\n");

    // Verify the z-scores have mean 0 and standard deviation 1
    println!("Z-score statistics:");
    println!(
        "- Mean: {:?}",
        stats::mean(&z_scores.view(), None).expect("Operation failed")
    );
    println!(
        "- Std Dev: {:?}",
        stats::std_dev(&z_scores.view(), None, 0).expect("Operation failed")
    );
}

// Function to standardize an array (convert to z-scores)
#[allow(dead_code)]
fn standardize_array(array: &Array2<f64>) -> Array2<f64> {
    // Calculate the global mean and standard deviation
    let mean = stats::mean(&array.view(), None).expect("Operation failed")[0];
    let std_dev = stats::std_dev(&array.view(), None, 0).expect("Operation failed")[0];

    // Apply the standardization formula: z = (x - mean) / std_dev
    array.mapv(|x| (x - mean) / std_dev)
}
