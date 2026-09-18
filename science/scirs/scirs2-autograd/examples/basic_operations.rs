use ag::tensor_ops::*;
use scirs2_autograd as ag;

#[allow(dead_code)]
fn main() {
    println!("Basic tensor operations example");

    // Create some tensors and perform basic operations
    ag::run(|g| {
        // Create tensors
        let a = convert_to_tensor(ag::ndarray::array![[1.0, 2.0], [3.0, 4.0]], g);
        let b = convert_to_tensor(ag::ndarray::array![[5.0, 6.0], [7.0, 8.0]], g);

        // Test basic operations
        let c = add(a, b);
        println!("a + b = {:?}", c.eval(g).expect("Operation failed"));

        let d = mul(a, b);
        println!("a * b = {:?}", d.eval(g).expect("Operation failed"));

        let e = sum_all(a);
        println!("sum(a) = {:?}", e.eval(g).expect("Operation failed"));

        let f = mean_all(b);
        println!("mean(b) = {:?}", f.eval(g).expect("Operation failed"));

        // Test matrix multiplication
        let b_t = transpose(b, &[1, 0]); // Transpose along the axes
        let g_tensor = matmul(a, b_t);
        println!(
            "a * b^T = {:?}",
            g_tensor.eval(g).expect("Operation failed")
        );

        // Test neural network operations
        let h = sigmoid(a);
        println!("sigmoid(a) = {:?}", h.eval(g).expect("Operation failed"));

        let i = relu(a);
        println!("relu(a) = {:?}", i.eval(g).expect("Operation failed"));

        let j = softmax(a, 1); // Softmax along axis 1
        println!("softmax(a) = {:?}", j.eval(g).expect("Operation failed"));

        println!("All operations completed successfully!");
    });
}
