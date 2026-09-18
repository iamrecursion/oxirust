extern crate scirs2_autograd as ag;
use ag::tensor_ops as T;

#[allow(dead_code)]
fn main() {
    println!("Testing shape preservation in tensor operations");

    // Test with scalar (0D) tensor
    ag::run(|ctx| {
        let scalar = T::scalar(5.0, ctx);

        println!("Scalar tensor:");
        scalar.show().eval(ctx).expect("Operation failed");

        // 1D tensor
        let vector = T::ones(&[3], ctx);

        println!("Vector tensor:");
        vector.show().eval(ctx).expect("Operation failed");

        // 2D tensor
        let matrix = T::ones(&[2, 2], ctx);

        println!("Matrix tensor:");
        matrix.show().eval(ctx).expect("Operation failed");

        // Test matrix operations
        let matrix_inv = T::matrix_inverse(matrix);

        println!("Matrix inverse:");
        matrix_inv.show().eval(ctx).expect("Operation failed");

        // Test determinant
        let det = T::determinant(matrix);

        println!("Determinant:");
        det.show().eval(ctx).expect("Operation failed");

        // Test adding a scalar to a matrix
        let added = matrix + scalar;

        println!("Matrix + Scalar:");
        added.show().eval(ctx).expect("Operation failed");

        // Test elementwise multiplication
        let multiplied = matrix * vector.reshape(&[3, 1]);

        println!("Matrix * Vector:");
        multiplied.show().eval(ctx).expect("Operation failed");

        println!("Test completed successfully");
    });
}
