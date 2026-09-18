extern crate scirs2_autograd as ag;

use ag::tensor_ops as T;

#[test]
#[allow(dead_code)]
fn test_basic_operations() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Basic scalar operations
        let a = T::scalar(3.0f32, ctx);
        let b = T::scalar(2.0f32, ctx);
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let div = a / b;

        // Test basic scalar arithmetic operations
        assert_eq!(sum.eval(ctx).expect("Test: operation failed")[[]], 5.0);
        assert_eq!(diff.eval(ctx).expect("Test: operation failed")[[]], 1.0);
        assert_eq!(prod.eval(ctx).expect("Test: operation failed")[[]], 6.0);
        assert_eq!(div.eval(ctx).expect("Test: operation failed")[[]], 1.5);

        // Verify that scalars have correct shape (0-dimensional)
        assert_eq!(
            a.eval(ctx).expect("Test: operation failed").shape(),
            &[] as &[usize]
        );
        assert_eq!(
            sum.eval(ctx).expect("Test: operation failed").shape(),
            &[] as &[usize]
        );
    });
}

#[test]
#[allow(dead_code)]
fn test_tensorshapes() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Test various tensor shapes
        let zeros = T::zeros(&[2, 3], ctx);
        let ones = T::ones(&[3, 2], ctx);

        // Test the actual tensor shapes directly
        let zeros_eval = zeros.eval(ctx).expect("Test: operation failed");
        let ones_eval = ones.eval(ctx).expect("Test: operation failed");

        println!("Zeros tensor shape: {:?}", zeros_eval.shape());
        println!("Ones tensor shape: {:?}", ones_eval.shape());

        // Verify tensor shapes directly
        assert_eq!(zeros_eval.shape(), &[2usize, 3usize]);
        assert_eq!(ones_eval.shape(), &[3usize, 2usize]);

        // Verify tensor contents
        assert!(zeros_eval.iter().all(|&x| x == 0.0));
        assert!(ones_eval.iter().all(|&x| x == 1.0));
    });
}
