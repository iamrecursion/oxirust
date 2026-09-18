// Example demonstrating the advanced ndarray operations

use scirs2_core::ndarray_ext::manipulation::{
    argmax, argmin, flip_2d, gradient, meshgrid, pad_2d, repeat_2d, roll_2d, tile_2d, unique,
};
use scirs2_core::ndarray_ext::{array, Array};

#[allow(dead_code)]
fn print_title(title: &str) {
    println!("\n{title}");
    println!("{}", "=".repeat(title.len()));
}

#[allow(dead_code)]
fn main() {
    println!("SciRS2-Core Advanced Array Operations Example");
    println!("=============================================\n");

    // Create some example arrays
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![3, 1, 2, 2, 3, 4, 1];
    let c = array![[5, 2, 3], [4, 1, 6]];

    // Demonstrate meshgrid
    print_title("Meshgrid");

    let x = array![1, 2, 3];
    let y = array![4, 5];

    let (x_grid, y_grid) = meshgrid(x.view(), y.view()).expect("Operation failed");
    println!("Input x-coordinates:");
    println!("{x}");
    println!("\nInput y-coordinates:");
    println!("{y}");
    println!("\nMeshgrid x_grid:");
    println!("{x_grid}");
    println!("\nMeshgrid y_grid:");
    println!("{y_grid}");

    // Demonstrate unique
    print_title("Unique Values");

    println!("Input array with duplicates:");
    println!("{b}");

    let unique_values = unique(b.view()).expect("Operation failed");
    println!("\nUnique values (sorted):");
    println!("{unique_values}");

    // Demonstrate argmin and argmax
    print_title("Finding Min/Max Indices");

    println!("Input array:");
    println!("{c}");

    // argmin examples
    let min_indices_cols = argmin(c.view(), Some(0)).expect("Operation failed");
    println!("\nIndices of minimum values in each column:");
    println!("{min_indices_cols}");

    let min_indices_rows = argmin(c.view(), Some(1)).expect("Operation failed");
    println!("\nIndices of minimum values in each row:");
    println!("{min_indices_rows}");

    let min_index = argmin(c.view(), None).expect("Operation failed");
    println!("\nIndex of overall minimum value (flattened):");
    println!("{}", min_index[0]);

    // argmax examples
    let max_indices_cols = argmax(c.view(), Some(0)).expect("Operation failed");
    println!("\nIndices of maximum values in each column:");
    println!("{max_indices_cols}");

    let max_indices_rows = argmax(c.view(), Some(1)).expect("Operation failed");
    println!("\nIndices of maximum values in each row:");
    println!("{max_indices_rows}");

    let max_index = argmax(c.view(), None).expect("Operation failed");
    println!("\nIndex of overall maximum value (flattened):");
    println!("{}", max_index[0]);

    // Demonstrate gradient
    print_title("Numerical Gradient");

    println!("Input array:");
    println!("{a}");

    let (grad_y, grad_x) = gradient(a.view(), None).expect("Operation failed");
    println!("\nGradient in y-direction (rows):");
    println!("{grad_y}");
    println!("\nGradient in x-direction (columns):");
    println!("{grad_x}");

    // With custom spacing
    let (grad_y, grad_x) = gradient(a.view(), Some((2.0, 0.5))).expect("Operation failed");
    println!("\nGradient with custom spacing (dy=2.0, dx=0.5):");
    println!("Gradient in y-direction (rows):");
    println!("{grad_y}");
    println!("Gradient in x-direction (columns):");
    println!("{grad_x}");

    // Demonstrate array manipulations
    print_title("Array Manipulations");

    println!("Original array:");
    println!("{a}");

    let flipped = flip_2d(a.view(), true, true);
    println!("\nFlipped along both axes:");
    println!("{flipped}");

    let rolled = roll_2d(a.view(), 1, 1);
    println!("\nRolled by 1 along both axes:");
    println!("{rolled}");

    let tiled = tile_2d(array![[1.0, 2.0], [3.0, 4.0]].view(), 2, 2);
    println!("\nTiled 2x2:");
    println!("{tiled}");

    let repeated = repeat_2d(array![[1.0, 2.0], [3.0, 4.0]].view(), 2, 2);
    println!("\nRepeated each element 2x2:");
    println!("{repeated}");

    let padded = pad_2d(a.view(), ((1, 1), (1, 1)), 0.0);
    println!("\nPadded with zeros:");
    println!("{padded}");

    // Complete example - calculating the magnitude of the gradient (like in image processing)
    print_title("Complete Example: Gradient Magnitude");

    // Create a sample 2D function (like an image)
    let img: Array<f64, _> = Array::from_shape_fn((5, 5), |(i, j)| {
        if i > 1 && i < 4 && j > 1 && j < 4 {
            10.0 // Higher value in the center
        } else {
            1.0 // Lower value at the edges
        }
    });

    println!("Sample 2D function (like an image):");
    println!("{img}");

    // Calculate gradient
    let (grad_y, grad_x) = gradient(img.view(), None).expect("Operation failed");

    println!("\nGradient in y-direction:");
    println!("{grad_y}");

    println!("\nGradient in x-direction:");
    println!("{grad_x}");

    // Compute gradient magnitude (sqrt(dx^2 + dy^2))
    let mut magnitude = Array::zeros(img.raw_dim());
    for i in 0..img.shape()[0] {
        for j in 0..img.shape()[1] {
            let gy = grad_y[[i, j]];
            let gx = grad_x[[i, j]];
            magnitude[[i, j]] = (gy * gy + gx * gx).sqrt();
        }
    }

    println!("\nGradient magnitude (edge detection):");
    println!("{magnitude}");

    // Find locations of strongest edges
    let max_val_idx = argmax(magnitude.view(), None).expect("Operation failed")[0];
    let rows = magnitude.shape()[1];
    let edge_row = max_val_idx / rows;
    let edge_col = max_val_idx % rows;

    println!("\nStrongest edge at position: ({edge_row}, {edge_col})");

    println!("\nExample complete!");
}
