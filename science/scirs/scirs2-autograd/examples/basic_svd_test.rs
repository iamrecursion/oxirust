use ag::tensor_ops::*;
use scirs2_autograd as ag;
use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn main() {
    println!("Testing basic SVD computation...");

    ag::run(|g| {
        // Create a simple 2x2 matrix
        let matrix_data = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix = variable(matrix_data.clone(), g);
        println!(
            "Original matrix shape: {:?}",
            matrix.eval(g).expect("Operation failed").shape()
        );

        // Compute SVD
        let (u, s, v) = svd(matrix);
        println!(
            "SVD shapes: U={:?}, S={:?}, V={:?}",
            u.eval(g).expect("Operation failed").shape(),
            s.eval(g).expect("Operation failed").shape(),
            v.eval(g).expect("Operation failed").shape()
        );

        // Print SVD components
        println!("U = {:?}", u.eval(g).expect("Operation failed"));
        println!("S = {:?}", s.eval(g).expect("Operation failed"));
        println!("V = {:?}", v.eval(g).expect("Operation failed"));

        // Verify reconstruction
        let s_diag = diag(s);
        let us = matmul(u, s_diag);
        let v_t = transpose(v, &[1, 0]);
        let reconstructed = matmul(us, v_t);

        println!(
            "Reconstructed = {:?}",
            reconstructed.eval(g).expect("Operation failed")
        );
        println!("Original = {:?}", matrix.eval(g).expect("Operation failed"));

        // Print numerical error
        let diff = sub(reconstructed, matrix);
        let error = sum_all(square(diff));
        println!(
            "Reconstruction error: {:?}",
            error.eval(g).expect("Operation failed")
        );
    });
}
